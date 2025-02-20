import pickle
import numpy as np
from keras.utils import Sequence
from tqdm import tqdm
from statsmodels import api as sm
import pandas as pd


def read_data_from_dataset(data_dir_path: str):
    """load train data and test data from the dataset

    Args:
        data_dir_path (str): path for the dataset we want to load (資料集所在的路徑)

    Returns:
        tuple: tuple which contains X_train, y_train, X_test and y_test in order
    """
    data_list = []
    for fname in ['X_train', 'y_train', 'X_test', 'y_test']:
        with open(f'{data_dir_path}/{fname}.pkl', 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    return tuple(data_list)


def generator(X: np.array, y: np.array, time_steps: int):
    """get time-series batch dataset

    Args:
        X (np.array): data for explanatory variables
        y (np.array): data for target variable
        time_steps (int): length of time series to consider during learning

    Returns:
        X_time (np.array): preprocessed data for explanatory variables
        y_time (np.array): preprocessed data for target variable

    """
    n_batches = X.shape[0] - time_steps - 1
    
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y_time = np.zeros((n_batches, 1))
    for i in range(n_batches):
        X_time[i] = X[i:(i + time_steps), :]
        y_time[i] = y[i + time_steps]
    return X_time, y_time


def split_dataset(X: np.array, y: np.array, ratio=0.8):
    """split dataset to train data and valid data in deep learning

    Args:
        X (np.array): data for explanatory variables
        y (np.array): data for target variable
        ratio (float, optional): ratio of train data and valid data. Defaults to 0.8.

    Returns:
        tuple: tuple which contains X_train, y_train, X_valid and y_valid in order
    """
    '''split dataset to train data and valid data'''
    X_train = X[:int(X.shape[0] * ratio)]
    y_train = y[:int(y.shape[0] * ratio)]
    X_valid = X[int(X.shape[0] * ratio):]
    y_valid = y[int(y.shape[0] * ratio):]
    dataset = tuple([X_train, y_train, X_valid, y_valid])

    return dataset


class ReccurentTrainingGenerator(Sequence): # 在訓練LSTM模型時生成批次的時序數據。
    """ Reccurent レイヤーを訓練するためのデータgeneratorクラス (訓練Recurrent層的數據生成器類別) """
    def _resetindices(self): # 隨機打亂數據的索引，以生成不同的批次數據。
        """バッチとして出力するデータのインデックスを乱数で生成する (作為批次輸出的數據索引用隨機數生成) """
        self.num_called = 0  # 同一のエポック内で __getitem__　メソッドが呼び出された回数 (在同一個epoch中調用 __getitem__ 方法的次數) # 用來計數在當前 epoch 中已經調用的批次數。
        
        all_idx = np.random.permutation(np.arange(self.num_samples)) #  生成隨機排列的索引
        remain_idx = np.random.choice(np.arange(self.num_samples),
                                      size=(self.steps_per_epoch * self.batch_size - len(all_idx)),
                                      replace=False)  # 足らない分を重複indexで補う (用重複的索引補足不足的部分) # 並填充不足部分的索引，使得每個epoch內的批次數與steps_per_epoch相符。
        self.indices = np.hstack([all_idx, remain_idx]).reshape(self.steps_per_epoch, self.batch_size) # 最終生成的索引數組，將所有批次的索引組合在一起。
        
    def __init__(self, x_set, y_set, batch_size, timesteps, delay):
        """
        x_set     : 説明変数 (データ点数×特徴量数)のNumPy配列
        y_set     : 目的変数 (データ点数×1)のNumPy配列
        batch_size: バッチサイズ
        timesteps : どの程度過去からデータをReccurent層に与えるか (決定要從多遠的過去數據提供給 Recurrent 層)
        delay     : 目的変数をどの程度遅らせるか (決定目標變數要延遲多長時間)
        """
        self.x = np.array(x_set) # 特徵數據
        self.y = np.array(y_set) # 標籤數據
        self.batch_size = batch_size # 批次大小
        self.steps = timesteps # 時間步數，即RNN模型輸入過去多少步的數據。
        self.delay = delay # 延遲步數，用於決定輸出的目標值相對於輸入的偏移量。
        
        self.num_samples = len(self.x) - timesteps - delay + 1 # 樣本數
        self.steps_per_epoch = int(np.ceil(self.num_samples / float(batch_size)))
        
        self._resetindices()
        
    def __len__(self): # 返回每個epoch中的批次數（步數），即steps_per_epoch。
        """ 1エポックあたりのステップ数を返す (返回每個epoch的步數) """ 
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ データをバッチにまとめて出力する (將數據整理成批次輸出) """
        indices_temp = self.indices[idx] # 當前批次的索引。
        
        batch_x = np.array([self.x[i:i+self.steps] for i in indices_temp]) # 根據 timesteps 生成的特徵數據，形狀為 (batch_size, timesteps, 特徵數)。
        batch_y = self.y[indices_temp + self.steps + self.delay - 1] # 延遲後的標籤數據，對應於batch_x最後一個時間步的預測目標。
        
        if self.num_called==(self.steps_per_epoch-1): # 在返回批次數據後，若已遍歷所有批次，則調用 _resetindices 隨機打亂索引，以便在下一 epoch 中生成不同的批次。
            self._resetindices() # 1エポック内の全てのバッチを返すと、データをシャッフルする (返回一個 epoch 中的所有批次後，將數據打亂順序)
        else:
            self.num_called += 1
        
        return batch_x, batch_y
    
    
class ReccurentPredictingGenerator(Sequence): # 生成遞歸神經網路（如 LSTM、GRU）模型的預測數據。
    """ Reccurent レイヤーで予測するためのデータgeneratorクラス (用於Recurrent層預測的數據生成器類別) """ 
    def __init__(self, x_set, batch_size, timesteps):
        """
        x_set     : 説明変数 (データ点数×特徴量数)のNumPy配列 (特徵數據的NumPy陣列，形狀為 (樣本數, 特徵數))
        batch_size: バッチサイズ (批次大小)
        timesteps : どの程度過去からデータをReccurent層に与えるか (時間步數，即每次預測所需的過去時間步數)
        """
        self.x = np.array(x_set) # 將輸入的數據轉換為 NumPy 陣列
        self.batch_size = batch_size # 批次大小
        self.steps = timesteps # 時間步數
        
        self.num_samples = len(self.x)-timesteps+1 # 計算可用的樣本數。每個樣本需要有timesteps的歷史數據。
        self.steps_per_epoch = int(np.floor(self.num_samples / float(batch_size))) # 每個epoch的步數
        
        self.idx_list = [] # 用於記錄批次索引
        
    def __len__(self):
        """ 1エポックあたりのステップ数を返す (返回每個epoch的步數，由樣本數和批次大小決定。) """
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ データをバッチにまとめて出力する (將數據整理成批次輸出) """
        start_idx = idx*self.batch_size # 計算當前批次的起始索引 start_idx。
        batch_x = [self.x[start_idx+i : start_idx+i+self.steps] for i in range(self.batch_size)] # 根據timesteps生成當前批次的數據，每個樣本包含self.steps個時間步。
        self.idx_list.append(start_idx) # 記錄當前批次的起始索引
        return np.array(batch_x) # 返回批次數據，形狀為(batch_size, timesteps, 特徵數)，適合用於 RNN 預測。


def decompose_time_series(x):
    
    step = len(x) // 10
    best_score = np.inf
    print('decomposing time series data ・・・・・')
    for period in tqdm(range(1, step + 1)):
        decompose_result = sm.tsa.seasonal_decompose(pd.Series(x), period=period, model='additive', extrapolate_trend='freq')
        print(len(np.where(decompose_result.resid < 0)[0]))
        score = np.sum(np.abs(decompose_result.resid))

        if score < best_score:
            best_period = period
            best_score = score
    print(f'best period : {best_period}')
    decompose_result = sm.tsa.seasonal_decompose(pd.Series(x), period=best_period, model='additive', extrapolate_trend='freq')

    x = {'trend': decompose_result.trend, 'period': decompose_result.seasonal, 'resid': decompose_result.resid}
    return x, best_period
