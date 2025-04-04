import random
import argparse # 解析命令列參數
import json
import os
from os import path, getcwd, makedirs, environ, listdir
import shutil
import numpy as np
import gc # Garbage Collector 垃圾回收機制

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 

    # from recombinator.optimal_block_length import optimal_block_length # 根據數據的特性，計算序列數據的最佳區塊長度（block length）。
    # ↑ 區塊長度的意義：區塊長度越長，數據的時間依賴性被保留得越多，但隨機性減少；區塊長度越短，數據更具隨機性，但可能失去時間依賴信息。
    # from recombinator.block_bootstrap import circular_block_bootstrap # 用於對具有時間依賴性的數據進行重抽樣，在不打破數據時間依賴性的情況下生成新的數據。採用的是循環抽樣的方式，這意味著當抽樣到序列尾端時，可以回到序列開頭繼續抽樣。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.cpp_extension

# 取得原始 include_paths()，避免遞迴呼叫自己
_original_include_paths = torch.utils.cpp_extension.include_paths

def include_paths_patched(*args, **kwargs):
    if 'cuda' in kwargs:
        del kwargs['cuda']  # 移除不支援的參數
    return _original_include_paths(*args, **kwargs)

torch.utils.cpp_extension.include_paths = include_paths_patched # 進行 Monkey Patch
print("✅ Patched torch.utils.cpp_extension.include_paths successfully!")

import xlstm  # 這行一定要放在 Monkey Patch 之後
from torchview import draw_graph # 模型結構視覺化

from utils.model import build_model, rmse, train_model
from utils.data_io import read_data_from_dataset
from utils.save import save_lr_curve, save_prediction_plot, save_yy_plot, save_mse, ResidualPlot, ErrorHistogram
from utils.device import limit_gpu_memory # 限制 TensorFlow 對 GPU 記憶體的預留或使用量。

from reports.Record_args_while_training import Record_args_while_training # 紀錄訓練時的nb_batch、bsize、period
from reports.Metrics_Comparison import metrics_comparison # 比較 Transfer-Learning遷移學習 vs. Without-Transfer-Learning不使用遷移學習
from reports.output import MSE_Improvement, MAE_Improvement # 比較 Transfer-Learning遷移學習 vs. Without-Transfer-Learning不使用遷移學習
from reports.util import dataset_idx_vs_improvement # 比較特徵非相似程度與MSE、MAE改進程度


def parse_arguments():
    ap = argparse.ArgumentParser(
        description='Time-Series Regression by LSTM through transfer learning') # 表示該程式的用途是透過遷移學習使用 LSTM 進行時間序列回歸分析。
    
    # for dataset path
    ap.add_argument('--out-dir', '-o', default='result',
                    type=str, help='path for output directory') # 指定輸出目錄的路徑，預設值為 result。
    
    # for model
    ap.add_argument('--seed', type=int, default=1234,
                    help='seed value for random value, (default : 1234)') # 確保隨機操作（如資料分割、模型初始化等）在每次執行中一致，方便實驗重現性。
    ap.add_argument('--train-ratio', default=0.8, type=float,
                    help='percentage of train data to be loaded (default : 0.8)') # 指定訓練集比例為 0.8（即 80%）。數據集會依據此比例分割為訓練集和測試集或驗證集。
    ap.add_argument('--time-window', default=1000, type=int,
                    help='length of time to capture at once (default : 1000)') # 設定時間窗口為 1000。這可能代表模型在一次處理過程中觀察的資料長度或時間範圍。
    
    # for training
    ap.add_argument('--train-mode', '-m', default='pre-train', type=str,
                    help='"pre-train", "transfer-learning", "without-transfer-learning", "comparison"\
                            "ensemble", "bagging", "noise-injection", "draw-model-graph", "score" (default : pre-train)') # 設定模式
    ap.add_argument('--gpu', action='store_true',
                    help='whether to do calculations on gpu machines (default : False)') # 是否啟用GPU加速
    ap.add_argument('--nb-epochs', '-e', default=1, type=int,
                    help='training epochs for the model (default : 1)') # 設定訓練的epoch。（epoch是完整地使用所有訓練數據訓練模型的一次過程。）
    ap.add_argument('--nb-batch', default=20, type=int,
                    help='number of batches in training (default : 20)') # 設定訓練過程中的批次數量，預設為 20。 批次大小（batch size） = 總訓練樣本數量 ÷ 批次數量（nb-batch）
    # ap.add_argument('--nb-subset', default=10, type=int,
    #                 help='number of data subset in bootstrapping (default : 10)') # 在bootstrapping中(即Bagging集成式學習)設定資料子集的數量。EX. 生成 10 個不同的訓練子集。
    ap.add_argument('--noise-var', default=0.0001, type=float,
                    help='variance of noise in noise injection (default : 0.0001)') # 在噪聲注入中設定噪聲的變異數。
    ap.add_argument('--valid-ratio', default=0.2, type=float,
                    help='ratio of validation data in train data (default : 0.2)') # 在訓練資料中設定驗證資料的比例。
    ap.add_argument('--freeze', action='store_true', 
                    help='whether to freeze transferred weights in transfer learning (default : False)') # 在遷移學習中凍結已轉移的權重。

    # for output
    ap.add_argument('--train-verbose', default=1, type=int,
                    help='whether to show the learning process (default : 1)') # 設定訓練過程中的輸出詳盡程度。
    args = vars(ap.parse_args())
    return args
    

def seed_every_thing(seed=1234): # 確保各種隨機操作（如資料分割、模型初始化等）在每次執行中產生相同的結果，從而提高實驗的可重現性。
    environ['PYTHONHASHSEED'] = str(seed) # 設定Python的雜湊隨機種子，確保Python的雜湊行為在每次執行時保持一致。
    np.random.seed(seed) # 設定NumPy的隨機種子，確保 NumPy 產生的隨機數在每次執行時相同。
    random.seed(seed) # 設定Python標準庫的隨機種子，確保Python標準庫中的隨機數生成器在每次執行時產生相同的結果。
    tf.random.set_seed(seed)  # TensorFlow 2.x 設定隨機種子的正確方式，確保TensorFlow產生的隨機數在每次執行時一致。


def save_arguments(args, out_dir): # 旨在將參數字典 args 以 JSON 格式保存到指定的輸出目錄 out_dir 中
    path_arguments = path.join(out_dir, 'params.json')
    with open(path_arguments, mode="w") as f:
        json.dump(args, f, indent=4)

# TODO: Delete
# def make_callbacks(file_path, save_csv=True):
#     # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=4, min_lr=1e-7) # 降低學習率，以促進模型更好地收斂。
#     model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True) # 保存最佳模型。 # -- save_weights_only = True,
#     # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) 
#     if not save_csv:
#         return [reduce_lr, model_checkpoint, early_stopping]
#     csv_logger = CSVLogger(path.join(path.dirname(file_path), 'epoch_log.csv')) # 將每個訓練週期的損失和評估指標記錄到 CSV 文件中
#     return [reduce_lr, model_checkpoint, csv_logger, early_stopping] 


def create_sliding_window(X, y, sequence_length):
    """
    使用滑動窗口方法，將時間序列轉換為 (batch_size, sequence_length, features) 格式。
    將原始的時間序列資料 (X) 切成「一小段一小段的序列片段」給 LSTM 使用，因為 LSTM 要吃的是 (batch_size, sequence_length, features) 格式的資料。
    :param X: 原始特徵數據 (samples, features)
    :param y: 標籤數據 (samples,)
    :param sequence_length: LSTM 所需的時間步長
    sequence_length=1440 代表我要用「前 1440 筆資料」去預測「第 1441 筆的值」。
    :return: 滑動窗口格式的 X 和 y
    """
    X_seq, y_seq = [], []
    total_samples = len(X)
    for i in range(total_samples - sequence_length): # for 迴圈切出每一組序列樣本
        X_seq.append(X[i: i + sequence_length])  # 取 sequence_length 長度的區間
        y_seq.append(y[i + sequence_length])  # 預測 sequence_length 之後的值
    return np.array(X_seq), np.array(y_seq)
 

def main():

    # make analysis environment
    limit_gpu_memory() #  限制TensorFlow在使用GPU時的記憶體佔用方式，避免因為分配過多而造成系統不穩定。然而當使用量超出設定的限制後，仍然可能發生OOM錯誤。
    args = parse_arguments() # 解析參數
    seed_every_thing(args["seed"]) # 設定隨機種子，在每次運行時產生一致的結果。
    write_out_dir = path.normpath(path.join(getcwd(), 'reports', args["out_dir"])) # 輸出文件的存放路徑
    makedirs(write_out_dir, exist_ok=True)
    
    print('-' * 140)
    print(f'train_mode: {args["train_mode"]} \n')
    
    if args["train_mode"] == 'pre-train': # 以預訓練模式執行模型訓練。
        
        for source in listdir('dataset/source'): # 逐個處理來源數據集

            # skip source dataset without pickle file
            data_dir_path = path.join('dataset', 'source', source)
            if not path.exists(f'{data_dir_path}/X_train.pkl'): continue
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], source) # 指定結果檔案(含模型)的保存路徑
            makedirs(write_result_out_dir, exist_ok=True)
            
            # load dataset
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path) # 讀取'X_train', 'y_train', 'X_test', 'y_test'資料
            X_train = np.concatenate((X_train, X_test), axis=0)  # > no need for test data when pre-training
            y_train = np.concatenate((y_train, y_test), axis=0)  # > no need for test data when pre-training
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 不隨機打亂數據 (shuffle=False)
            print(f'\nSource dataset : {source}')
            print(f"📌 X_train.shape: {X_train.shape}")  # 查看訓練數據形狀
            print(f"📌 X_valid.shape: {X_valid.shape}")  # 查看訓練數據形狀
            print(f'切分比例: {args["valid_ratio"]}')
            
            # construct the model
            sequence_length = 1440  # (period) 表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
            input_shape = (sequence_length, X_train.shape[1]) # (timesteps, features)，period表示時間步數，X_train.shape[1]為欄位特徵。
            input_dim = X_train.shape[1]  # 取得資料集的特徵數
            print(f'sequence_length:{sequence_length}, args["nb_batch"]: {args["nb_batch"]}')
            model, device = build_model(input_shape=(sequence_length, input_dim), gpu=True)

            # **進行訓練**
            # 重新塑形數據，使其符合 (samples, sequence_length, features)
            # ✅ 把原本的訓練和驗證資料轉換成適合 LSTM 的格式，確保 X_train 形狀正確。
            X_train_seq, y_train_seq = create_sliding_window(X_train, y_train, sequence_length=1440) # 創建訓練數據
            X_valid_seq, y_valid_seq = create_sliding_window(X_valid, y_valid, sequence_length=1440) # 創建驗證數據
            # ✅ 轉換為 PyTorch tensor 格式
            # 轉成 tensor 是為了讓模型能使用 GPU 加速訓練。
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32) # 創建訓練數據
            y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
            X_valid_tensor = torch.tensor(X_valid_seq, dtype=torch.float32) # 創建驗證數據
            y_valid_tensor = torch.tensor(y_valid_seq, dtype=torch.float32)
            # ✅ 建立 PyTorch DataLoader
            # 把輸入和對應的標籤包成一組，方便 DataLoader 抽樣。
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # 創建訓練數據
            val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor) # 創建驗證數據
            # ✅ 分批讀取資料
            # shuffle=False：不打亂順序（時間序列通常要保留時間順序）
            # drop_last=False：保留最後不足一整批的資料
            bsize = len(y_train) // args["nb_batch"] # 計算批次大小batch_size # --min
            print(f'計算批次大小batch_size: {bsize}')
            train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=False, drop_last=False) 
            val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False, drop_last=False) 
            
            # train the model
            print('開始訓練model模型（Pre-Train）')
            Record_args_while_training(write_out_dir, args["train_mode"], source, args['nb_batch'], bsize, sequence_length, data_size=(len(y_train) + len(y_test))) # 記錄參數
            model, train_loss, val_loss, optimizer = train_model(model, train_loader, val_loader, num_epochs=args["nb_epochs"], save_file_path=write_result_out_dir,
                                                                learning_rate=1e-4, device=device, early_stop_patience=10, monitor="val_loss")
            save_lr_curve(train_loss, val_loss, write_result_out_dir, source) # 保存每個epoch的學習曲線

            # prediction (進行預測並保存結果) 使用Testing資料試著預測。
            model.eval() # 確保模型處於評估模式
            X_test_seq, y_test_seq = create_sliding_window(X_test, y_test, sequence_length=1440)  # 創建時序窗口
            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device) # 轉換 X_test 為 PyTorch tensor
            # 進行推理
            with torch.no_grad():
                y_test_pred = model(X_test_tensor)
            y_test_pred = y_test_pred.cpu().numpy() # 轉換為 numpy 陣列以供繪圖
           
            # save log for the model (計算誤差指標並保存結果) 
            y_test = y_test_seq # y_test_seq 已經跟 X_test_seq 對齊了，是 sliding window 對應的 ground truth。 #?? # y_test_pred -> 有問題
            save_prediction_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
            save_yy_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
            mse_score, rmse_loss, mae_loss, mape_loss, msle_loss, r2 = save_mse(y_test, y_test_pred, write_result_out_dir, model=model, sequence_length=sequence_length, input_dim=input_dim) # 計算y_test和y_test_pred之間的均方誤差（MSE）分數，同時將模型摘要資訊寫入文件。
            args["MAE Loss"] = mae_loss
            args["MSE Loss"] = mse_score
            args["RMSE Loss"] = rmse_loss
            args["MAPE Loss"] = mape_loss
            args["MSLE Loss"] = msle_loss
            args["R2 Score"] = r2
            Learning_Rate = optimizer.param_groups[0]["lr"] # 取得學習率
            args["Learning Rate"] = Learning_Rate
            save_arguments(args, write_result_out_dir) # 保存訓練參數 (args) 到結果輸出目錄中。
            ResidualPlot(y_test, y_test_pred, write_result_out_dir)
            ErrorHistogram(y_test, y_test_pred, write_result_out_dir)

            # clear memory up (清理記憶體並保存參數)
            del model # 刪除舊模型
            gc.collect() # 清理 CPU 記憶體
            torch.cuda.empty_cache()  # 清空CUDA記憶體緩存
            print('\n' * 2 + '-' * 140 + '\n' * 2)
    

    elif args["train_mode"] == 'transfer-learning': # 使用遷移學習來訓練模型，從預訓練模型中提取權重並應用於新數據集。
        
        for target in listdir('dataset/target'):
        
            # skip target in the absence of pickle file
            if not path.exists(f'dataset/target/{target}/X_train.pkl'): continue

            for source in listdir(f'{write_out_dir}/pre-train'): # 遍歷預訓練的模型，對每個模型進行遷移學習。
                
                pre_model_path = f'{write_out_dir}/pre-train/{source}/best_model.pt' # 預訓練模型的最佳權重 (.pt 檔案)
                if not path.exists(pre_model_path): continue # 確保預訓練模型權重存在。

                # make output directory 保存結果的目錄
                if args["freeze"]:
                    print(f'在遷移學習中，是否凍結權重: {args["freeze"]}，即凍結權重。')
                    train_mode = f'{args["train_mode"]} (Freeze)'
                else:
                    print(f'在遷移學習中，是否凍結權重: {args["freeze"]}，即解凍權重。')
                    train_mode = f'{args["train_mode"]} (Unfreeze)'
                write_result_out_dir = path.join(write_out_dir, train_mode, target, source)
                makedirs(write_result_out_dir, exist_ok=True)
                    
                # load dataset (加載目標數據集)
                data_dir_path = f'dataset/target/{target}'
                X_train, y_train, X_test, y_test = \
                    read_data_from_dataset(data_dir_path)
                # === 1️⃣ 定義模型輸入形狀（time steps, features）===
                sequence_length = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。 
                input_shape = (sequence_length, X_train.shape[1]) # (timesteps, features)，period表示時間步數，X_train.shape[1]為欄位特徵。
                input_dim = X_train.shape[1]  # 取得資料集的特徵數
                X_train, X_valid, y_train, y_valid = \
                    train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 將訓練集分割為訓練和驗證。
                print(f'\nTarget dataset : {target}')
                print(f'\nSource dataset : {source}')
                print(f'\nX_train.shape : {X_train.shape}')
                print(f'\nX_valid.shape : {X_valid.shape}')
                print(f'\nX_test.shape : {X_test.shape}')
                print(f'sequence_length:{sequence_length}, args["nb_batch"]: {args["nb_batch"]}')
                
                # construct the model (構建並編譯模型)
                # train the model (訓練模型)
                # 重新塑形數據，使其符合 (samples, sequence_length, features)
                # ✅ 把原本的訓練和驗證資料轉換成適合 LSTM 的格式，確保 X_train 形狀正確。
                X_train_seq, y_train_seq = create_sliding_window(X_train, y_train, sequence_length=1440) # 創建訓練數據
                X_valid_seq, y_valid_seq = create_sliding_window(X_valid, y_valid, sequence_length=1440) # 創建驗證數據
                # ✅ 轉換為 PyTorch tensor 格式
                # 轉成 tensor 是為了讓模型能使用 GPU 加速訓練。
                X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32) # 創建訓練數據
                y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
                X_valid_tensor = torch.tensor(X_valid_seq, dtype=torch.float32) # 創建驗證數據
                y_valid_tensor = torch.tensor(y_valid_seq, dtype=torch.float32)
                # ✅ 建立 PyTorch DataLoader
                # 把輸入和對應的標籤包成一組，方便 DataLoader 抽樣。
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # 創建訓練數據
                val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor) # 創建驗證數據
                # ✅ 分批讀取資料
                # shuffle=False：不打亂順序（時間序列通常要保留時間順序）
                # drop_last=False：保留最後不足一整批的資料
                print(f'開始建立模型（{args["train_mode"]}）')
                bsize = len(y_train) // args["nb_batch"] # 計算批次大小batch_size # --min
                print(f'計算批次大小batch_size: {bsize}')
                train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=False, drop_last=False) 
                val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False, drop_last=False) 
                pre_model, _ = build_model(input_shape=input_shape, gpu=args["gpu"], verbose=False) # 1️⃣ 先建立模型架構
                pre_model.load_state_dict(torch.load(pre_model_path)) # 2️⃣ 載入預訓練的模型權重
                print(f"✅ 成功載入預訓練模型權重：{pre_model_path}")
                model, device = build_model(input_shape=input_shape, gpu=args["gpu"], pre_model=pre_model, freeze=args["freeze"], verbose=True) # 3️⃣ 傳入 build_model() 做 Transfer Learning  # freeze參數決定是否凍結預訓練模型的層，以避免在遷移學習中微調它們。
        
                # train the model (訓練模型)
                print(f'開始訓練model模型（{args["train_mode"]}）')
                Record_args_while_training(write_out_dir, train_mode, target, args['nb_batch'], bsize, sequence_length, data_size=(len(y_train) + len(y_valid) + len(y_test)))
                model, train_loss, val_loss, optimizer = train_model(model, train_loader, val_loader, num_epochs=args["nb_epochs"], save_file_path=write_result_out_dir,
                                        learning_rate=1e-4, device=device, early_stop_patience=10, monitor="val_loss") # 訓練模型
                save_lr_curve(train_loss, val_loss, write_result_out_dir, target) # 繪製學習曲線
                
                # prediction (進行預測並保存結果)
                # === 載入最佳模型 ===
                best_model = build_model(input_shape=(sequence_length, input_dim), gpu=True)[0]  # 初始化模型
                file_path = path.join(write_result_out_dir, f'best_model.pt')
                best_model.load_state_dict(torch.load(file_path))  # 載入訓練好的模型權重
                best_model.eval()  # 設定為評估模式
                # === 創建時序視窗（測試集）=== 
                print("📌 開始對測試集進行推論")
                X_test_seq, y_test_seq = create_sliding_window(X_test, y_test, sequence_length=sequence_length)  # 創建時序窗口，把 X_test 切成 (samples, timesteps, features)。
                # X_test_seq → 預測用的測試資料（切成 time series 視窗）
                # y_test_seq → 對應的 ground truth（你要評估模型的真實標籤）
                X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
                test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
                test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=1, # 保持小批次，減少一次佔用記憶體
                                            shuffle=False,
                                            pin_memory=True, # 開啟資料固定於主記憶體，加快 GPU 傳輸速度（尤其在 CUDA）
                                            num_workers=0 # 減少併發讀取，避免耗RAM。
                                            )
                # === 執行預測 === 
                y_test_pred = []
                with torch.no_grad():  # 不計算梯度
                    for (x_batch,) in test_loader: # 每次從 test_loader 拿出一筆 (batch_size=1)
                        x_batch = x_batch.to("cuda" if torch.cuda.is_available() else "cpu") # 將資料送入 GPU or CPU
                        pred = best_model(x_batch) # 執行預測（forward），這一行就是預測的核心。
                        y_test_pred.append(pred.cpu().numpy()) # 預測結果轉為numpy，加入列表中
                y_test_pred = np.concatenate(y_test_pred, axis=0)  # 合併所有預測結果

                # save log for the model (計算MSE並保存結果)
                y_test = y_test_seq # y_test_seq 已經跟 X_test_seq 對齊了，是 sliding window 對應的 ground truth。 #?? # y_test_pred -> 有問題
                save_prediction_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
                save_yy_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
                # 計算各種回歸指標
                mse_score, rmse_loss, mae_loss, mape_loss, msle_loss, r2 = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model, sequence_length=sequence_length, input_dim=input_dim) # 計算y_test和y_test_pred之間的均方誤差（MSE）分數，同時將模型摘要資訊寫入文件。
                args["MAE Loss"] = mae_loss
                args["MSE Loss"] = mse_score
                args["RMSE Loss"] = rmse_loss
                args["MAPE Loss"] = mape_loss
                args["MSLE Loss"] = msle_loss
                args["R2 Score"] = r2
                Learning_Rate = optimizer.param_groups[0]["lr"] # 取得學習率
                args["Learning Rate"] = Learning_Rate
                save_arguments(args, write_result_out_dir) # 保存本次訓練或測試的所有參數設定及結果。
                ResidualPlot(y_test, y_test_pred, write_result_out_dir)
                ErrorHistogram(y_test, y_test_pred, write_result_out_dir) # 誤差直方圖

                # clear memory up (清理記憶體並保存參數)
                del model, best_model # 刪除舊模型
                gc.collect() # 清理 CPU 記憶體
                torch.cuda.empty_cache()  # 清空CUDA記憶體緩存
                print('\n' * 2 + '-' * 140 + '\n' * 2)
    

    elif args["train_mode"] == 'without-transfer-learning': # 不使用遷移學習

        for target in listdir('dataset/target'):
        
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset (加載數據集並分割為訓練和驗證集)
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path) # 讀取'X_train', 'y_train', 'X_test', 'y_test'資料
            sequence_length = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 不隨機打亂數據 (shuffle=False)
            print(f'\nTarget dataset : {target}')
            print(f'\nX_train shape: {X_train.shape}')
            print(f'\nX_valid shape: {X_valid.shape}')
            print(f'\nX_test shape: {X_test.shape}')
            print(f'sequence_length:{sequence_length}, args["nb_batch"]: {args["nb_batch"]}')
            
            # construct the model (構建模型)
            input_shape = (sequence_length, X_train.shape[1])
            input_dim = X_train.shape[1]  # 取得資料集的特徵數
            model, device = build_model(input_shape=(sequence_length, input_dim), gpu=True)
            
            # train the model (訓練模型)
            # 重新塑形數據，使其符合 (samples, sequence_length, features)
            # ✅ 把原本的訓練和驗證資料轉換成適合 LSTM 的格式，確保 X_train 形狀正確。
            X_train_seq, y_train_seq = create_sliding_window(X_train, y_train, sequence_length=1440) # 創建訓練數據
            X_valid_seq, y_valid_seq = create_sliding_window(X_valid, y_valid, sequence_length=1440) # 創建驗證數據
            # ✅ 轉換為 PyTorch tensor 格式
            # 轉成 tensor 是為了讓模型能使用 GPU 加速訓練。
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32) # 創建訓練數據
            y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
            X_valid_tensor = torch.tensor(X_valid_seq, dtype=torch.float32) # 創建驗證數據
            y_valid_tensor = torch.tensor(y_valid_seq, dtype=torch.float32)
            # ✅ 建立 PyTorch DataLoader
            # 把輸入和對應的標籤包成一組，方便 DataLoader 抽樣。
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # 創建訓練數據
            val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor) # 創建驗證數據
            # ✅ 分批讀取資料
            # shuffle=False：不打亂順序（時間序列通常要保留時間順序）
            # drop_last=False：保留最後不足一整批的資料
            bsize = len(y_train) // args["nb_batch"] # 計算批次大小batch_size # --min
            print(f'計算批次大小batch_size: {bsize}')
            train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=False, drop_last=False) 
            val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False, drop_last=False) 
            print(f'開始訓練model模型（{args["train_mode"]}）')
            Record_args_while_training(write_out_dir, args["train_mode"], target, args['nb_batch'], bsize, sequence_length, data_size=(len(y_train) + len(y_valid) + len(y_test)))
            model, train_loss, val_loss, optimizer = train_model(model, train_loader, val_loader, num_epochs=args["nb_epochs"], save_file_path=write_result_out_dir,
                                                    learning_rate=1e-4, device=device, early_stop_patience=10, monitor="val_loss")
            save_lr_curve(train_loss, val_loss, write_result_out_dir, target) # 繪製學習曲線

            # prediction (預測)
            # === 載入最佳模型 ===    # best_model = load_model(file_path, custom_objects={'rmse': rmse}) # 傳遞rmse自定義指標
            best_model = build_model(input_shape=(sequence_length, input_dim), gpu=True)[0]  # 初始化模型
            file_path = path.join(write_result_out_dir, "best_model.pt")
            best_model.load_state_dict(torch.load(file_path))  # 載入訓練好的模型權重
            best_model.eval()  # 設定為評估模式
            # === 創建時序視窗（測試集）===       # RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period) # 生成測試數據。
            print("📌 開始對測試集進行推論")
            X_test_seq, y_test_seq = create_sliding_window(X_test, y_test, sequence_length=sequence_length)  # 創建時序窗口，把 X_test 切成 (samples, timesteps, features)。
            # X_test_seq → 預測用的測試資料（切成 time series 視窗）
            # y_test_seq → 對應的 ground truth（你要評估模型的真實標籤）
            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
            test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
            test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                      batch_size=1, # 保持小批次，減少一次佔用記憶體
                                                      shuffle=False,
                                                      pin_memory=True, # 開啟資料固定於主記憶體，加快 GPU 傳輸速度（尤其在 CUDA）
                                                      num_workers=0 # 減少併發讀取，避免耗RAM。
                                                      )
            # === 執行預測 ===    # y_test_pred = best_model.predict_generator(RPG) # 預測測試數據
            y_test_pred = []
            with torch.no_grad():  # 不計算梯度
                for (x_batch,) in test_loader: # 每次從 test_loader 拿出一筆 (batch_size=1)
                    x_batch = x_batch.to("cuda" if torch.cuda.is_available() else "cpu") # 將資料送入 GPU or CPU
                    pred = best_model(x_batch) # 執行預測（forward），這一行就是預測的核心。
                    y_test_pred.append(pred.cpu().numpy()) # 預測結果轉為numpy，加入列表中
            y_test_pred = np.concatenate(y_test_pred, axis=0)  # 合併所有預測結果

            # save log for the model (計算MSE誤差和保存結果)
            y_test = y_test_seq # 直接用對應過的 ground truth #?? # y_test_pred -> 有問題
            # print(f'y_test_pred: {y_test_pred}') 
            # print(f'y_test: {y_test}')
            save_prediction_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
            save_yy_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
            # 計算各種回歸指標
            mse_score, rmse_loss, mae_loss, mape_loss, msle_loss, r2 = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model, sequence_length=sequence_length, input_dim=input_dim) # 計算y_test和y_test_pred之間的均方誤差（MSE）分數，
            args["MAE Loss"] = mae_loss
            args["MSE Loss"] = mse_score
            args["RMSE Loss"] = rmse_loss
            args["MAPE Loss"] = mape_loss
            args["MSLE Loss"] = msle_loss
            args["R2 Score"] = r2
            Learning_Rate = optimizer.param_groups[0]["lr"] # 取得學習率
            args["Learning Rate"] = Learning_Rate
            save_arguments(args, write_result_out_dir) # 保存本次訓練或測試的所有參數設定及結果。
            ResidualPlot(y_test, y_test_pred, write_result_out_dir)
            ErrorHistogram(y_test, y_test_pred, write_result_out_dir) # 誤差直方圖

            # clear memory up (清理記憶體)
            del model, best_model # 刪除舊模型
            gc.collect() # 清理 CPU 記憶體
            torch.cuda.empty_cache()  # 清空CUDA記憶體緩存
            print('\n' * 2 + '-' * 140 + '\n' * 2)
    
    
    elif args["train_mode"] == 'comparison': # 比較 Transfer-Learning遷移學習 vs. Without-Transfer-Learning不使用遷移學習
        out_dir, train_mode = write_out_dir, args["train_mode"]
        metrics_comparison(out_dir, train_mode) # 比較所有metrics。
        MSE_Improvement(out_dir, train_mode) # 比較MSE
        MAE_Improvement(out_dir, train_mode) # 比較MAE
        dataset_idx_vs_improvement(out_dir, train_mode) # 'DTW'
    
    
    elif args["train_mode"] == 'ensemble': # 使用ensemble整體學習。通過聚合、以平均的方式來得到最終預測結果
        
        from Ensemble import start_ensemble # 整體學習 
        
        for target in listdir('dataset/target'):    
            
            # make output directory
            TL_model_dir = path.join(write_out_dir, args["train_mode"], target, 'model')
            makedirs(TL_model_dir, exist_ok=True) # 建立目標目錄（如果不存在） 
            
            # 將transfer-learning (Unfreeze)的模型複製搬移到ensemble底下的model資料夾
            source_dir = path.join(write_out_dir,'transfer-learning (Unfreeze)', target) # 獲取模型來源資料夾
            for TL_model in listdir(source_dir):
                src_path = path.join(source_dir, TL_model, f'best_model.pt')
                if path.isfile(src_path): # 檢查是否是檔案再執行複製
                    new_model_name = f"{TL_model}_best_model.pt" # 新的檔名為「來源名稱_best_model.pt」
                    dest_path = path.join(TL_model_dir, new_model_name)
                    shutil.copy(src_path, dest_path) # 複製檔案到目標資料夾（覆蓋既有檔案）
                    print(f"已複製: {src_path} -> {dest_path}")
            print("模型複製完成。")
                    
            # ensemble整體學習 預測與評估。
            period = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
            start_ensemble (period, write_out_dir=path.join(write_out_dir, args["train_mode"]))
    
            # clear memory up 清理記憶體
            gc.collect() # 清理 CPU 記憶體
            torch.cuda.empty_cache()  # 清空CUDA記憶體緩存
    
            print('\n' * 2 + '-' * 140 + '\n' * 2)        
        
    
    # elif args["train_mode"] == 'noise-injection': # 添加隨機噪聲來訓練模型，使模型在訓練過程中遇到更多的數據變化，減少過擬合並提高模型對測試數據的泛化能力。

    #     for target in listdir('dataset/target'):
            
    #         # make output directory (設置輸出目錄)
    #         write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
    #         makedirs(write_result_out_dir, exist_ok=True)

    #         # load dataset (加載數據集並切分為訓練和驗證集)
    #         data_dir_path = path.join('dataset', 'target', target)
    #         X_train, y_train, X_test, y_test = \
    #             read_data_from_dataset(data_dir_path)
    #         period = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
    #         X_train, X_valid, y_train, y_valid =  \
    #             train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 將訓練數據劃分為訓練集和驗證集。
    #         print(f'\nTarget dataset : {target}')
    #         print(f'\nX_train : {X_train.shape}')
    #         print(f'\nX_valid : {X_valid.shape}')
    #         print(f'\nX_test : {X_test.shape[0]}')

    #         # construct the model
    #         file_path = path.join(write_result_out_dir, 'best_model.hdf5')
    #         callbacks = make_callbacks(file_path)
    #         input_shape = (period, X_train.shape[1])
    #         model = build_model(input_shape, args["gpu"], write_result_out_dir, noise=args["noise_var"])

    #         # train the model
    #         bsize = len(y_train) // args["nb_batch"]
    #         RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1) # 生成訓練數據，以批次形式提供給模型。
    #         RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1) # 生成驗證數據，以批次形式提供給模型。
    #         Record_args_while_training(write_out_dir, args["train_mode"], target, args['nb_batch'], bsize, period, data_size=(len(y_train) + len(y_valid) + len(y_test)))
    #         H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks) # 訓練模型
    #         save_lr_curve(H, write_result_out_dir, target) # 繪製學習曲線

    #         # prediction
    #         best_model = load_model(file_path)
    #         RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period) # 生成測試數據。
    #         y_test_pred = best_model.predict_generator(RPG) # 預測測試數據

    #         # save log for the model
    #         y_test = y_test[-len(y_test_pred):] # 將y_test的長度調整為與 y_test_pred（模型預測值）的長度一致，確保在進行計算和可視化時，兩者長度相符。
    #         save_prediction_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
    #         save_yy_plot(y_test, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
    #         mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model) # 計算y_test和y_test_pred之間的均方誤差（MSE）分數，
    #         args["mse"] = mse_score
    #         save_arguments(args, write_result_out_dir) # 保存本次訓練或測試的所有參數設定及結果。

    #         # clear memory up (清理記憶體)
    #         keras.backend.clear_session()
    #         print('\n' * 2 + '-' * 140 + '\n' * 2)


    elif args["train_mode"] == 'draw-model-graph':
    # === 模型結構視覺化 ===
        # make output directory
        write_result_out_dir = path.join(write_out_dir, args["train_mode"])
        makedirs(write_result_out_dir, exist_ok=True)

        sequence_length = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
        input_dim = 5 # X_train.shape[1] → 取得資料集的特徵數
        input_shape = (sequence_length, input_dim)  # 時序長度 1440、特徵數量 5
        model, _ = build_model(input_shape=input_shape, gpu=False, verbose=False)

        graph = draw_graph(model, input_size=(1, *input_shape), expand_nested=True,
                    graph_name="xLSTMModel", roll=True, save_graph=True,
                    directory=write_result_out_dir, filename="xLSTM_architecture")
        
        graph.visual_graph.render(format='png')  # 儲存成 PNG 檔
        print(f"模型結構圖已儲存至: {os.path.join(write_result_out_dir, 'xLSTM_architecture.png')}")
    

    else:
        print('No matchining train_mode')

if __name__ == '__main__':
    main()
