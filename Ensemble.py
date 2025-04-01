import os
import gc
from utils.data_io import read_data_from_dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm # 顯示進度條，看到任務的完成進度。
import numpy as np
from utils.save import save_prediction_plot, save_yy_plot, save_mse, ResidualPlot, ErrorHistogram

from main import create_sliding_window # 代替舊的 ReccurentPredictingGenerator
from utils.model import build_model # 直接用 PyTorch 架構初始化模型並載入 .pt 權重。


def ensemble (sequence_length, write_out_dir, target):
    '''
    ensemble集成式學習的預測結果
    '''
    print(f'\n目標資料集: {target}')
    write_result_out_dir = os.path.join(write_out_dir, target)
    print(f'Ensemble 預測輸出路徑: {write_result_out_dir}')

    # === 讀取測試資料 ===
    data_dir_path = os.path.join('.', 'dataset', 'target', target)
    _, _, X_test, y_test = read_data_from_dataset(data_dir_path)

    if not sequence_length:
        sequence_length = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
    print(f'使用時間步長（sequence_length）: {sequence_length}')

    # === 建立滑動視窗 ===
    X_test_seq, y_test_seq = create_sliding_window(X_test, y_test, sequence_length)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, # 保持 batch=1，就等同以前的 RPG 逐筆處理
        shuffle=False,
        pin_memory=True, # 將 DataLoader 載入的資料放到固定的記憶體（pinned memory）中，這樣 資料能更快傳送到 GPU。
        num_workers=0 # 所有資料讀取在 主執行緒 中完成（單執行緒）【最安全】
    )

    # === 對每個模型進行推論 ===
    prediction = [] # 用於存儲每個模型的預測值
    # 加載模型並生成預測
    model_dir = os.path.join(write_result_out_dir, 'model')
    for model_file in tqdm(os.listdir(model_dir), desc='Loading Models'): # 為 model_dir 裡的每個模型檔案建立一個 tqdm 進度條，顯示文字為 Loading Models
        model_path = os.path.join(model_dir, model_file)
        print(f'使用模型: {target} の {model_file} \n路徑: {model_path}')

        # 初始化與載入模型
        input_shape = (sequence_length, X_test.shape[1])
        model, device = build_model(input_shape=input_shape, gpu=True, verbose=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 預測
        y_test_pred = []
        with torch.no_grad():
            for (x_batch,) in test_loader: # test_loader => 使用 PyTorch 的 DataLoader 逐筆推論，節省記憶體
                x_batch = x_batch.to(device)
                pred = model(x_batch)
                y_test_pred.append(pred.cpu().numpy())
        y_test_pred = np.concatenate(y_test_pred, axis=0)
        prediction.append(y_test_pred)

        # 手動清理模型以釋放顯存
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # === 集成平均 ===
    # 計算預測結果的準確度，並保存所有的預測數據。集成所有預測結果。
    prediction = np.array(prediction) # 轉換為NumPy陣列，形狀為 (模型數量, 測試數據大小, 預測特徵數=1)。 # 單變量預測結果
    print(f'\n所有模型預測 shape: {prediction.shape}') # prediction shape: (模型數量, 測試樣本數, 1)
    mean_pred = prediction.squeeze() # # 去掉最後一個維度。EX.將數據形狀從 (3, 31942, 1) 轉為 (3, 31942)
    mean_pred = np.mean(mean_pred, axis=0) # 對每行數據取平均值作為集成預測結果，即集成預測值。 # (n_samples,)

    y_test = y_test_seq # === 確保 ground truth 對齊 ===   # ??
    # size_test = prediction.shape[1] # 測試數據的樣本數量。
    # y_test[-size_test:] # 將y_test修正為與size_test長度相同，確保測試標籤與預測數據對齊。
    print(f'y_test.shape: {y_test.shape}, pred.shape: {mean_pred.shape}') # 與y_test的形狀一致
        
    # 將預測輸出彙總成CSV保存 # === 保存各模型預測 & 平均值 ===
    df = pd.DataFrame() # 組合所有預測結果為 DataFrame
    for i in range(prediction.shape[0]):
        df[f'Prediction_{i+1}'] = prediction[i].flatten() # 將每一組預測結果展平為一列
    df['Average'] = mean_pred # 預測平均值
    df.to_csv( os.path.join(write_result_out_dir, 'combined_predictions.csv'), index=False) # 保存到同一個CSV文件            
    np.save( os.path.join(write_result_out_dir, f'{target}_ensemble_pred.npy'), prediction) # 將所有的預測結果保存為NumPy檔案(.npy)，可用np.load讀取資料。

    # 計算預測誤差 # === 評估與繪圖 ===
    mse_score, rmse_loss, mae_loss, mape_loss, msle_loss, r2 = save_mse(y_test, mean_pred, write_result_out_dir) # 計算y_test和mean_pred之間的均方誤差（MSE）分數，同時將模型摘要資訊寫入文件。
    # 繪製圖表
    plt.rcParams['font.size'] = 25 # 設定字體大小
    plt.figure(figsize=(15, 7)) # 建立圖表
    save_prediction_plot(y_test, mean_pred, write_result_out_dir) # 繪製y_test與mean_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
    save_yy_plot(y_test, mean_pred, write_result_out_dir) # 繪製y_test與mean_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
    ResidualPlot(y_test, mean_pred, write_result_out_dir)
    ErrorHistogram(y_test, mean_pred, write_result_out_dir)
    print(f'\n✅ 集成預測完成！MSE: {mse_score:.4f}, MAE: {mae_loss:.4f}, R²: {r2:.4f}\n')


def start_ensemble (sequence_length, write_out_dir):
    folders = [f for f in os.listdir(write_out_dir) if os.path.isdir(os.path.join(write_out_dir, f))]
    for target in folders:
        ensemble (sequence_length, write_out_dir, target)
        torch.cuda.empty_cache()
        gc.collect()
        print('\n' * 2 + '-' * 140 + '\n' * 2)
    print('✅ Ensemble Learning 全部結束 🎉')