import os
import numpy as np
import pandas as pd

# TODO: Delete
# import tensorflow as tf
# from keras.layers import Input, Dense, BatchNormalization
# from keras.layers import TimeDistributed # wrappers
# from keras.layers import GaussianNoise # noise
# from keras.models import Model
# from keras.utils import plot_model
# from keras.optimizers import Adam
# from keras import initializers, regularizers
# import keras.backend as K

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# 取得原始 include_paths()，避免遞迴呼叫自己
_original_include_paths = torch.utils.cpp_extension.include_paths
def include_paths_patched(*args, **kwargs):
    if 'cuda' in kwargs:
        del kwargs['cuda']  # 移除不支援的參數
    return _original_include_paths(*args, **kwargs)
torch.utils.cpp_extension.include_paths = include_paths_patched # 進行 Monkey Patch
print("✅ Patched torch.utils.cpp_extension.include_paths successfully!")

import xlstm
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig  # 匯入xLSTM所需的類別
from xlstm.blocks.slstm.block import sLSTMBlockConfig  # sLSTM設定
from xlstm.blocks.slstm.block import sLSTMBlock


class LinearHeadwiseExpandConfig:
    '''
    封裝模型的「輸入特徵數量」與「頭數（num_heads）」的組合設定，
    把一個輸入向量分成多個head處理。
    "把原本的特徵向量切成 N 份，分別用不同的 Linear 處理後再拼起來"
    "輸入向量 → 切成 N 份 → 各跑自己 Linear → 拼起來"
    '''
    def __init__(self, in_features, num_heads):
        self.in_features = in_features # 輸入的特徵數（通常是embedding維度）。
        self.num_heads = num_heads # 要分成幾個「頭」來並行運算。
        # print(f"🔍 Debug: LinearHeadwiseExpandConfig - in_features={self.in_features}, num_heads={self.num_heads}") # 輸出目前的設定值
        # 把輸入特徵切成 num_heads 個部分，每個 head 要處理相同長度的向量。
        assert self.in_features % self.num_heads == 0, \
            f"⚠️ AssertionError: in_features ({self.in_features}) 必須是 num_heads ({self.num_heads}) 的倍數" # 檢查 in_features 必須可以整除 num_heads，否則會報錯！


class sLSTMBlockConfig:
    '''
    定義一個名為 sLSTMBlockConfig 的設定類別（configuration class），
    類似於一種“參數封裝器”，方便模型其他地方引用和傳遞設定值。
    '''
    def __init__(self, num_heads=8):  # 假設預設 num_heads=8
        self.num_heads = num_heads
        self.slstm = self  # 確保slstm屬性可存取
        # print(f"🔍 Debug: sLSTMBlockConfig - num_heads={self.num_heads}")
    def __post_init__(self):
        """這個方法是 `xLSTMBlockStackConfig` 期望調用的 `__post_init__()` 方法"""
        # print(f"🔧 Debug: `sLSTMBlockConfig.__post_init__()` 被調用")


class xLSTMBlockStack(nn.Module): 
    '''
    模型的堆疊單元，管理多個sLSTMBlock。
    負責建立一層一層堆疊的 sLSTMBlock 模型（即多層 LSTM 結構）。
    : config.num_blocks = 幾層 sLSTMBlock
    : config.slstm = 每層的設定（sLSTMBlockConfig）
    '''
    def __init__(self, config: xLSTMBlockStackConfig):
        '''
        : config.num_blocks 會是希望堆幾層 LSTM block。
        : config.slstm_block 是每一層 block 的設定（通常是 sLSTMBlockConfig 的實例）。
        '''
        super(xLSTMBlockStack, self).__init__()
        self.blocks = self._create_blocks(config=config) # 讀取 config.slstm_at(也就是index)，逐層判斷要放哪個 block。
        # print(f"🔍 Debug: 在 xLSTMBlockStack 內: embedding_dim={config.embedding_dim}")

    def _create_blocks(self, config):
        '''
        每一層 block（LSTM or sLSTM）是怎麼被決定和建立的，
        最後把所有 block 串成一個模型堆疊結構。
        : config.num_blocks 控制要建立幾層 block。
        '''
        blocks = []
        for i in range(config.num_blocks): # -- 目前只有 sLSTMBlock，預設是都使用它，每層都用同一份設定。
            if i in config.slstm_at: 
                # config.slstm_at =>「指派哪幾層用 sLSTM」。
                # _create_blocks() 讀取設定，決定要堆哪些 Block。
                print(f"✅ Block {i}: 使用 sLSTMBlock")
                blocks.append(sLSTMBlock(config=config.slstm_block)) # 可以創建多個sLSTMBlock。
            else:
                print(f"🧱 Block {i}: 使用預設 LSTMBlock（或其他）")
                print("!!! 當前尚未調整 !!! 目前仍為 sLSTMBlock，可自行替換。")
                blocks.append(sLSTMBlock(config=config.slstm_block))  # 可以改放其他 block 類型，例如 LSTMBlock。
                # blocks.append(LSTMBlock(input_dim=config.embedding_dim)) # -- 新增 LSTMBlock 類別
        return nn.ModuleList(blocks) # 確保blocks是nn.ModuleList。 # 用nn.ModuleList包裝起來，用來儲存多個子模型，確保它們能被訓練、存檔、轉移到 GPU。
        # ! 【↑問題】這段雖然會疊多層 block，但它每一層都只用同一個 sLSTMBlock，沒有根據 slstm_at 來判斷是否該使用 sLSTMBlock 還是別的 block（如 LSTMBlock）。
        # ! ↑ 這樣就可以根據 slstm_at = [1] 的設定，只在第 1 層使用 sLSTMBlock，其餘使用其他 block（目前仍是 sLSTMBlock）。
    
    def forward(self, x):
        """
        前向傳播時逐層通過 block，「疊加」發生的地方。
        :param x: (batch_size, sequence_length, embedding_dim)
        :return: (batch_size, sequence_length, embedding_dim)
        """
        for block in self.blocks: # blocks 是 list of block (例如：block0, block1, block2)
            x = block(x)  # 每個 block 接收上層的輸出，逐層計算。 \
                          # 執行完一層後，x會更新，再丟進下一層。
        return x


class sLSTMBlock(nn.Module): 
    '''
    核心計算單元，整個xLSTM架構中最底層的模型運算單位，負責對每一段輸入序列做LSTM處理。
    '''
    
    def __init__(self, config):
        '''
        初始化 sLSTMBlock。
        : input_size=5 每個時間步的輸入特徵數（即 embedding 維度）
        : hidden_size=5 LSTM的輸出維度
        : num_layers=1 單層LSTM
        : batch_first=True 輸入格式為 (batch_size, sequence_length, features)
        '''
        super(sLSTMBlock, self).__init__()
        self.num_heads = config.num_heads # sLSTMBlock使用的「多頭數」。
        # 這裡可以加入具體的 LSTM 層結構
        self.lstm_layer = nn.LSTM(input_size=5, # 輸入特徵數
                                  hidden_size=5, # LSTM 的輸出維度
                                  num_layers=1, # 單層
                                  batch_first=True # 輸入格式為 (batch, seq, feature)
                                  )

    def forward(self, x):
        """
        前向傳播
        :param x: (batch_size, sequence_length, embedding_dim)
        :return x: (batch_size, sequence_length, embedding_dim)
        :return _: (h_n, c_n)，也就是 LSTM 的最終 hidden state 和 cell state（這裡沒用到）。
        """
        x, _ = self.lstm_layer(x)  # 把序列 x 丟進 LSTM，輸出每個時間步的特徵（不是只取最後一步喔）。
        return x # 回傳經過LSTM處理後的結果，對一段序列套用LSTM處理的結果。 # 每個時間步都經過變換，代表有更深層的時序理解
    

class LSTMBlock(nn.Module):
    def __init__(self, input_dim):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class xLSTMModel(nn.Module): # 創建xLSTM
    def __init__(self, input_dim=5, output_dim=1, context_length=60, num_blocks=2, embedding_dim=5, use_slstm=False):
        """
        PyTorch 版 xLSTM 模型，定義了 xLSTMBlockStack 作為主體。
        :param input_dim: 輸入特徵維度
        :param output_dim: 輸出維度
        :param context_length: 訓練時長
        :param num_blocks: xLSTM block 數量
        :param embedding_dim: LSTM 內部特徵維度
        :param use_slstm: 是否啟用 sLSTM
        """
        super(xLSTMModel, self).__init__()

        slstm_config = sLSTMBlockConfig() if use_slstm else None # 如果 use_slstm = True，就會建立一個 sLSTMBlockConfig 實例。 # -- num_heads=8
        if slstm_config and hasattr(slstm_config, "num_heads"): # 如果啟用了 sLSTM，並且有num_heads，則檢查 num_heads
            # hasattr(obj, "attr") => 用來檢查某個物件 (obj) 是否 有某個屬性 (attr)。
            print(f"🔍 Debug: sLSTMBlockConfig.num_heads = {slstm_config.num_heads}")
            pass
        else:
            print(f"⚠️ Warning: sLSTMBlockConfig 沒有 `num_heads` 屬性，請檢查其定義！")
            pass

        # 建立 xLSTMBlockStackConfig
        self.xlstm_config = xLSTMBlockStackConfig(
            context_length=context_length,
            num_blocks=num_blocks, # 有幾層 block（num_blocks）
            embedding_dim=embedding_dim, # 每層的輸入維度是多少（embedding_dim），設定為5。
            slstm_block=sLSTMBlockConfig() if use_slstm else None, # 用什麼樣的 sLSTM 配置(num_heads)
            slstm_at=[1] if use_slstm else [],  # 哪幾層使用 sLSTMBlock。這裡是 index=1，讓第二個 block 使用 sLSTM。 
                                                # 可以靈活選擇哪幾層要用進階的 sLSTMBlock，其他層就保留為普通的 LSTMBlock 或預設 block 結構。
        )
        # print(f"🔍 Debug: 在 xLSTMModel 中: input_dim={input_dim}, embedding_dim={embedding_dim}, num_heads={num_heads}")

        self.xlstm_stack = xLSTMBlockStack(self.xlstm_config) # 將剛剛建立好的 xLSTMBlockStackConfig 配置，傳入 xLSTMBlockStack 做實例化，逐層判斷要放哪個 block，建立一個「多層堆疊的 xLSTM block」堆疊體。
        self.batch_norm = nn.BatchNorm1d(embedding_dim) # 加入BatchNormalization批次正規化，幫助穩定訓練。 # ! 因為作用在 (batch, features, time)，所以要 permute() 兩次。
        self.fc = nn.Linear(embedding_dim, output_dim) # 全連接層(fc)，最後只取 最後一個時間步的輸出 → 做線性轉換 → 輸出預測值。

    def forward(self, x):
        """
        前向傳播
        :param x: 輸入張量 (batch_size, sequence_length, input_dim=5)
        :return: 預測輸出
        """
        x = self.xlstm_stack(x)  # 輸入序列進入堆疊的 block。
        # self.xlstm_stack(x) 這一行是模型的「堆疊 feature extractor」。
        # 它會根據設定，自動把 x 送進好幾層 LSTM 或 sLSTM block 裡面。
        # 每層會更新特徵，讓模型有更深層次的理解能力。

        # running_mean should contain 60 elements not 5
        # 以下三行是為了正確使用 BatchNorm1d，它要求 (batch, channels, seq_len) 的輸入格式。
        x = x.permute(0, 2, 1)  # 變成 (batch_size, embedding_dim, sequence_length)
        x = self.batch_norm(x)   # 透過批次正規化，使訓練更穩定
        x = x.permute(0, 2, 1)   # 變回 (batch_size, sequence_length, embedding_dim)
        x = self.fc(x[:, -1, :])  # 全連接層輸出，取最後一個時間步的輸出來做預測。
        return x
    

class EarlyStopping:
    """ 
    監測驗證Loss，若連續多次無改善則提前終止訓練，防止模型過擬合或浪費資源持續訓練沒在進步的模型。
    當驗證指標連續幾個 epoch 沒有「足夠的改善」，就提前中止訓練，並回復「最佳模型權重」。
    """
    def __init__(self, patience=10, min_delta=0.0001, monitor="val_loss", verbose=True, restore_best_weights=True):
        """
        :param patience: 經過多少個epoch後沒有改善
        :param min_delta: 最小改善幅度。如果指標（如 val_loss 或 val_rmse）的改善幅度小於 min_delta，則不視為有效改善。
        :param monitor: 監測的指標 ('val_loss' 或 'val_rmse')
        :param verbose: 是否打印EarlyStopping訊息
        :param restore_best_weights: 是否在 EarlyStopping觸發時回復最佳模型權重
        """
        self.patience = patience
        self.min_delta = min_delta # 最小改善幅度。必須進步夠多，才值得延長訓練。
        self.monitor = monitor
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_score = float('inf') # 記錄最佳指標
        self.counter = 0 # 沒進步的次數計數器
        self.early_stop = False # 是否提前停止
        self.best_model_state = None  # 用於存儲最佳模型權重

    def __call__(self, score, model):
        """
        :param model: 目前的 PyTorch 模型
        :param score: 監測的數值（可以是 val_loss 或 val_rmse）
        # :param val_loss: 驗證損失 (MSE)
        # :param val_rmse: 驗證 RMSE (可選)
        """
        # TODO: Delete
        # if self.monitor == "val_loss":
        #     score = val_loss
        # elif self.monitor == "val_rmse":
        #     score = val_rmse
        # else:
        #     raise ValueError("monitor 參數只能是 'val_loss' 或 'val_rmse'")

        # 如果當前分數比之前最佳的還要好，則更新最佳分數並重置patience計數。
        if score < self.best_score - self.min_delta: # 而且改善幅度大於 min_delta
            self.best_score  = score # 更新 best_score
            self.counter = 0  # 重置 patience 計數 （因為有進步）
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()  # 保存最佳模型權重
        else: # 沒有進步
            self.counter += 1 # 沒有明顯進步就把 counter 加 1
            if self.verbose:
                print(f"⏳ EarlyStopping patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience: # 如果 counter 累積超過耐心（patience）次數
                self.early_stop = True  # 啟動 EarlyStopping
                print(f"⏹ Early stopping triggered after {self.patience} epochs of no improvement.") # 耐心計數：每次指標沒有改善時，打印當前的耐心計數，方便調試。
            if self.restore_best_weights and self.best_model_state: # 如果啟用了 restore_best_weights，則回復最佳權重。
                print("🔄 Restoring best model weights...")
                model.load_state_dict(self.best_model_state) # 回復最佳模型權重：確保最終的模型不是來自過擬合的 epoch。


# 自訂 RMSE 函數
def rmse(y_true, y_pred): # 因為Keras並未內建RMSE作為指標，需要自行定義一個自訂的RMSE指標函數。
    '''
    Root Mean Squared Error (RMSE)
    RMSE 是 mse 的平方根，更直觀地表示誤差，與實際數據單位一致。
    '''
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


# TODO: 訓練模型函數
def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-4, device="cuda", save_file_path=None, early_stop_patience=10, monitor="val_loss"):
    # out_dir=None,
    """
    使用PyTorch訓練xLSTM模型，並記錄Loss以供繪製學習曲線，同時記錄每個epoch的指標至CSV。
    :param model: xLSTMModel
    :param train_loader: 訓練數據加載器
    :param val_loader: 驗證數據加載器
    :param num_epochs: 訓練週期
    :param learning_rate: 學習率
    :param device: 運行設備 ('cuda' or 'cpu')
    """
    model.to(device) # 將模型搬到 GPU 或 CPU
    print(f'Model Device: {next(model.parameters()).device}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 定義 optimizer (選擇 Adam 最佳化器)。
    criterion = nn.MSELoss() # 損失函數，使用MSE作為損失函數。

    # 存訓練紀錄與早停準備
    train_loss_list = []  # 記錄訓練 Loss
    val_loss_list = []  # 記錄驗證 Loss

    # 學習率調整策略 (當驗證 loss 連續6次沒有改善，學習率減少) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True, min_lr=1e-7) # 當驗證集 (val_loss) 連續6次 沒有改善時，學習率會縮小 50% (factor=0.5)。
    early_stopping = EarlyStopping(patience=early_stop_patience, monitor=monitor, verbose=True) # 若 val_loss 持續無改善 → 終止訓練
    
    # 追蹤最佳模型
    best_val_loss = float('inf') # 設定一個極大值（無限大）來初始化，這樣第一個 epoch 的 val_loss 一定會「比它小」，就會被視為最佳。
    best_model_state = None # 用來存放「最佳模型的參數權重」（也就是 model.state_dict()）。

    log_file = os.path.join(save_file_path, "epoch_log.csv") # 建立 CSV 紀錄檔案，記錄訓練歷程。
    log_columns =  ["epoch", "loss", "lr", "mae", "mse", "rmse", "val_loss", "val_mae", "val_mse", "val_rmse"]
    log_df = pd.DataFrame(columns=log_columns) # 建立一個空表格，準備每個 epoch 結束後寫入一行資料（訓練結果）。

    #  TODO: 訓練迴圈（每個 epoch）
    for epoch in range(num_epochs):
        # 清梯度 → 預測 → 計算Loss → 反向傳播 → 更新參數
        model.train() # 設定為訓練模式
        train_loss = 0.0 # 訓練損失，代表模型在「訓練集」上的誤差，是模型每個 epoch 在訓練資料上的整體損失平均。
        mae_train = 0.0  #追蹤MAE
        mse_train = 0.0 #追蹤MSE

        for inputs, targets in train_loader: # 從train_loader讀入資料
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() # 清除梯度
            outputs = model(inputs) # 前向傳播。把 inputs 資料送進模型 model 中，並得到模型的輸出 outputs。
            loss = criterion(outputs, targets) # 計算損失
            loss.backward() # 反向傳播（分析錯在哪）。從 loss 開始，反向傳遞誤差，計算每個參數的梯度（gradient）。
                            # 梯度 => 就是如果要讓錯誤變小，這個參數應該往哪個方向調整。
            optimizer.step() # 更新權重（根據分析結果來更新權重）。
                             # 依照「梯度方向」調整參數，「學習 → 修改參數 → 讓模型在下一次做得更好！」。
            train_loss += loss.item()
            mse_train += loss.item() #計算MSE
            mae_train += torch.abs(outputs - targets).mean().item()  #計算MAE

        # 全部批次結束後，平均 loss、mae、mse
        train_loss /= len(train_loader)
        mse_train /= len(train_loader)
        rmse_train = mse_train ** 0.5
        mae_train /= len(train_loader)

        # TODO: 驗證，計算 Validation Loss。
        model.eval()  # 切換為驗證模式
        val_loss = 0.0 # 驗證損失，代表模型在「驗證集」上的誤差，是模型在看過訓練資料之後，驗證在沒看過的資料上是否泛化良好。
        mse_val = 0.0 # 追蹤MSE
        mae_val = 0.0  # 追蹤MAE
        with torch.no_grad(): # 停止自動計算梯度，不會多花資源做梯度計算。
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) # 前向傳播
                loss = criterion(outputs, targets) # 計算損失
                val_loss += loss.item()
                mse_val += loss.item() # 計算MSE
                mae_val += torch.abs(outputs - targets).mean().item()  # 計算MAE
        # 平均 loss、mae、mse
        val_loss /= len(val_loader)
        mse_val /= len(val_loader)
        rmse_val = mse_val ** 0.5
        mae_val /= len(val_loader)

        # 更新學習率
        scheduler.step(val_loss)

        # 儲存最佳模型
        if val_loss < best_val_loss: # 記錄訓練過程中最好的模型狀態，當每一個 epoch 結束時，如果新的驗證損失變得更小，就會更新。
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        # TODO: EarlyStopping 停止訓練
        score = val_loss if early_stopping.monitor == "val_loss" else rmse_val
        early_stopping(score, model)
        if early_stopping.early_stop:
            print(f"⏹ 訓練提前終止於 Epoch {epoch+1}")
            break

        # ✅ 記錄 Loss
        train_loss_list.append(train_loss)  # 修正為 train_loss
        val_loss_list.append(val_loss)  # 直接記錄 val_loss
        current_lr = optimizer.param_groups[0]["lr"] # 取得當前學習率
        # 記錄Epoch資料到DataFrame。會把每一輪的結果寫入epoch_log.csv，追蹤 Loss、RMSE 的變化曲線 
        epoch_data = pd.DataFrame({
            "epoch": [epoch+1], # 當前訓練週期（從 1 開始）
            "loss": [train_loss], # 訓練損失（MSE）
            "lr": [current_lr], # 學習率
            "mae": [mae_train], # 訓練的 MAE（平均絕對誤差）
            "mse": [mse_train], # 訓練 MSE
            "rmse": [rmse_train], # 訓練 RMSE
            "val_loss": [val_loss], # 驗證損失
            "val_mae": [mae_val], # 驗證 MAE
            "val_mse": [mse_val], # 驗證 MSE
            "val_rmse": [rmse_val] # 驗證 RMSE
        })
        log_df = pd.concat([log_df, epoch_data], ignore_index=True) # 這行會把剛剛的 epoch_data（一行資料）加進整個表格 log_df 中的最後一行。
        log_df.to_csv(log_file, index=False) # 寫入CSV。這是「每一輪都寫一次」，所以即使中途中斷，檔案裡也會保留紀錄。 
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val RMSE: {rmse_val:.6f}")
    
    # === 模型訓練結束 ===
    # 讓模型回到最佳狀態
    if early_stopping.restore_best_weights and best_model_state is not None:
        model.load_state_dict(best_model_state) # 回復到那個「表現最好的模型」。
        print("訓練完成，已恢復最佳模型狀態")
    # 儲存最佳模型權重到檔案
    if save_file_path:
        best_model_path = os.path.join(save_file_path, "best_model.pt") # 儲存為 PyTorch 格式 (.pt)
        torch.save(best_model_state, best_model_path)
        print(f"✅ 最佳模型權重已儲存至: {best_model_path}")

    return model, train_loss_list, val_loss_list, optimizer # 回傳最終模型


# TODO: 建立模型結構
def build_model(input_shape: tuple, # 模型的輸入形狀(timesteps, features)
                gpu=True, # 若為True則使用CUDA加速的LSTM層（CuDNNLSTM）
                pre_model=None, # 若有傳入預訓練模型，則可以從中載入權重。
                freeze=False, # 若為True，會將部分層設為不可訓練，用於遷移學習。
                noise=None, # 若設定此參數，會加入一層高斯噪聲層，模擬數據變異。
                verbose=True, # 是否印出模型架構
                ):
    """
    建立 xLSTM 模型
    根據參數來組裝、設定、甚至載入預訓練模型。
    """
    # 設置 GPU
    # 有 GPU 可用 (torch.cuda.is_available()) 且 允許使用 (gpu=True) => device = "cuda"
    # 否則 => device = "cpu"
    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    print(f"運行設備: {device}")

    # 參數
    input_dim = input_shape[1]  # 特徵數量 = 5
    output_dim = 1  # 預測輸出維度
    sequence_length = input_shape[0]  # 時序長度

     # 創建 xLSTM 模型 (初始化模型)
    model = xLSTMModel(
        input_dim=input_dim,
        output_dim=output_dim,
        context_length=sequence_length,
        num_blocks=2,
        embedding_dim=input_dim, # 特徵維度，設為5，並讓embedding_dim = input_dim。
        use_slstm=True, # 是否啟用 sLSTM block
    ).to(device)

    # 加載預訓練權重（如果有）
    if pre_model: # 實現 Transfer Learning 或續訓的關鍵，把之前訓練好的模型權重載進來使用。
        model.load_state_dict(pre_model.state_dict())
        model.to(device)  # 確保權重載入後還是對應到 device
        print("成功載入預訓練模型權重")

    # 設置權重是否可訓練
    if freeze: # 凍結權重
        for param in model.parameters():
            param.requires_grad = False # 是 PyTorch 中控制「這個參數是否參與訓練」的設定。
        print("所有層已凍結，模型將不會更新權重")

    # 打印模型資訊
    if verbose: # 顯示完整模型結構、每層輸出大小、參數量
        summary(model, input_size=(1, sequence_length, input_dim), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]) # print(model)

    return model, device


    
    print(f'是否加入噪聲: {noise}')
    if noise:
        noise_input = GaussianNoise(np.sqrt(noise))(input_layer) # 加入高斯噪聲層，用於模擬數據的隨機變異。屬於正則化技術，而非數據擴充（Data Augmentation）。np.sqrt(noise) 表示噪聲的標準差。
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
                kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性，並且有效減少梯度消失或梯度爆炸問題。
                bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
            )
        )(noise_input)

    else:
        dense = TimeDistributed(
            Dense( 
                10, # 輸出單元數為10的全連接層。
                kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
                kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
                bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
            )
        )(input_layer)

    lstm1 = LSTM(
        60,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        recurrent_initializer=initializers.Orthogonal(seed=0), # 將 LSTM 的遞歸權重初始化為正交矩陣，以促進梯度穩定。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(dense)
    lstm1 = BatchNormalization()(lstm1) # 正規化，穩定訓練過程、加速收斂，並提高模型的泛化能力。

    lstm2 = LSTM(
        60,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        recurrent_initializer=initializers.Orthogonal(seed=0), # 將 LSTM 的遞歸權重初始化為正交矩陣，以促進梯度穩定。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(lstm1)
    lstm2 = BatchNormalization()(lstm2) # 正規化，穩定訓練過程、加速收斂，並提高模型的泛化能力。

    output_layer = Dense(
        1,
        activation='sigmoid', # 激活函數為sigmoid，適合輸出一個範圍在0到1之間的預測結果。
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(lstm2)

    model = Model(inputs=input_layer, outputs=output_layer) # 建立模型
    if savefig:
        plot_model(model, to_file=f'{write_result_out_dir}/architecture.png', show_shapes=True, show_layer_names=True) # 繪製神經網路模型的結構並將其保存為圖片檔案
    

    # transfer weights from pre-trained model (遷移學習：載入預訓練模型權重)
    if pre_model:
        for i in range(2, len(model.layers) - 1): #（跳過輸入層和最後輸出層）           
            print(f"\n--- Layer {i}: {model.layers[i].name} ---")

            # 獲取當前層的原始權重
            original_weights = model.layers[i].get_weights() # 模型當前層的初始權重。
            # print(f"Original Weights (Before):\n{original_weights}")
            
            # 將對應層的權重從預訓練模型載入
            pre_trained_weights = pre_model.layers[i].get_weights() # 從預訓練模型中加載的權重。
            # print(f"Pre-trained Weights (Loaded):\n{pre_trained_weights}")
            
            # 獲取更新後的權重            
            model.layers[i].set_weights(pre_model.layers[i].get_weights()) # 將對應層的權重從預訓練模型載入。

            # 獲取更新後的權重
            updated_weights = model.layers[i].get_weights() # 載入預訓練權重後的權重。
            # print(f"Updated Weights (After):\n{updated_weights}")

            # 檢查是否與預訓練模型一致
            if all(np.array_equal(w1, w2) for w1, w2 in zip(updated_weights, pre_trained_weights)):
                print("Weights successfully updated to pre-trained weights!")
            else:
                print("Weights mismatch after update!")

            if freeze: # 若 freeze=True，則將這些層設置為不可訓練（即權重不會在訓練中更新），這樣可以保持預訓練權重不變。
                model.layers[i].trainable = False
                print(f"Layer {i} ({model.layers[i].name}) is now frozen and will not be updated during training.")
            else: # 否則，權重可訓練。允許模型在新數據上學習特定模式。
                model.layers[i].trainable = True # 保證層被設置為可訓練（防止之前被凍結）
                print(f"Layer {i} ({model.layers[i].name}) is trainable and its weights will be updated during training (fine-tuning).")

    # 調整優化器&學習率。
    if pre_model:
        # 定義每個數據集的學習率。預訓練模型的微調通常需要更小的學習率。
        dataset_learning_rates = {
            'FishAquaponics_IoTpond2': 1e-5,  # 針對 IoTpond2
            'FishAquaponics_IoTpond3': 1e-4,  # 針對 IoTpond3
            'FishAquaponics_IoTpond4': 1e-4,  # 針對 IoTpond4
        }
        current_dataset = write_result_out_dir.split(os.sep)[-1]  # 根據系統路徑分隔符分割路徑，取得最後一個路徑部分。
        init_learning_rate = dataset_learning_rates.get(current_dataset, 1e-4)  # 默認學習率為 1e-4
    else:
        # 其它
        dataset_learning_rates = {
            'FishAquaponics_IoTpond3': 1e-5,  # 針對 IoTpond3
        }
        current_dataset = write_result_out_dir.split(os.sep)[-1]  # 根據系統路徑分隔符分割路徑，取得最後一個路徑部分。
        init_learning_rate = dataset_learning_rates.get(current_dataset, 1e-4)  # 默認學習率為 1e-4，適合大多數模型的初始訓練。
    print(f'初始學習率: {init_learning_rate}')
    Adam_optimizer = Adam(learning_rate=init_learning_rate) # 標準Adam優化器
    print(f'優化器參數: {Adam_optimizer.get_config()}')
    model.compile(optimizer=Adam_optimizer, loss='mse', metrics=['mse', rmse, 'mae','mape','msle']) # metrics是模型訓練過程中用來監控模型性能的指標、評估模型的訓練效果。 # 使用動態學習率調整策略（如 ReduceLROnPlateau），設定初始學習率有助於更好地控制學習率範圍。
    if verbose: print(model.summary())

    return model
    
'''
GaussianNoise 高斯噪聲層
好處：
1.) 防止過擬合：模型會學習到更多數據的變異性，而不是過度擬合訓練數據的特定模式。
2.) 增加泛化能力：模型在訓練時遇到更多不同的數據輸入，從而提高模型在測試數據上的性能。
3.) 模擬數據噪聲：幫助模型在面對真實世界的噪聲數據時表現得更好。

L2 正則化 
目的：減少過度擬合，也就是讓模型不過於貼合訓練數據，使它能對新數據有更好的表現。
原理：L2 正則化會讓模型的權重（即每個輸入對預測結果的影響程度）保持較小，以便更簡單、更平滑地適應數據。
作法：L2 正則化通過在損失函數中加入一項與權重的平方和成比例的懲罰項來抑制模型的複雜度，從而減少過度擬合風險。
如何實現：它會把每個權重值的平方乘上一個小係數（如 0.01），然後把這些結果加到損失函數中。這會讓模型更偏好小權重，從而減少過度擬合的風險。

Orthogonal 初始化
目的：初始化模型權重，使模型在一開始訓練時更穩定。
原理：Orthogonal 初始化會讓模型的初始權重彼此間保持「正交」，也就是說，它們的方向完全不相關。這樣能確保訊號在傳遞過程中不會過於減弱或變得過強，這對於 RNN 等時間序列模型特別有幫助。
如何實現：在模型開始訓練前，為權重分配一組特殊的初始值，使它們的方向是彼此「垂直」的（數學上稱之為「正交矩陣」）。

Glorot 均勻初始化方法 是什麼？
目的：確保模型一開始的權重不會讓數據信號在傳遞過程中爆炸或消失。
原理：Glorot 均勻初始化（也叫 Xavier 初始化）會根據輸入與輸出的神經元數量，選擇一個適當的範圍，從中均勻分布地選取權重值。這樣可以保持訊號穩定，避免模型在訓練初期就出現梯度爆炸或消失的問題。
如何實現：模型的權重會隨機從 [-limit, limit] 的範圍中選取，其中 limit 是基於該層的輸入和輸出單元數計算得來的。

BatchNormalization (批次正規化) 是什麼？
目的：加速訓練，並讓模型更穩定。
原理：批次正規化會在每一層的輸出上進行「標準化」，也就是把輸出調整成一個更穩定的範圍（通常平均值為 0，標準差為 1）。這樣做可以讓模型更快找到最佳解，減少訓練過程中的波動。
如何實現：模型會根據當前的批次數據，計算出每層輸出的平均值和標準差，然後把每個輸出都調整到該範圍內。這樣做可以平衡輸出的大小，提升模型的穩定性和收斂速度。
'''
