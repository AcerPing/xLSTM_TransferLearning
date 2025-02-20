from os import path, listdir
from utils.data_io import (
    read_data_from_dataset,
    ReccurentPredictingGenerator
)
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import keras.backend as K
from tqdm import tqdm # 顯示進度條，看到任務的完成進度。
import numpy as np
from utils.save import save_prediction_plot, save_yy_plot, save_mse, ResidualPlot, ErrorHistogram


# 自訂RMSE函數
def rmse(y_true, y_pred): # 因為Keras並未內建RMSE作為指標，需要自行定義一個自訂的RMSE指標函數。
    '''
    RMSE 是 mse 的平方根，更直觀地表示誤差，與實際數據單位一致。
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def ensemble (period, write_out_dir, target):
    '''
    ensemble集成式學習的預測結果
    '''
    print(target)
    write_result_out_dir = path.join(write_out_dir, target)
    print(f'Ensemble Learning output directory: {write_result_out_dir}')

    # load dataset (載入數據集)
    data_dir_path = path.join('.', 'dataset', 'target', target)
    X_train, y_train, X_test, y_test = \
        read_data_from_dataset(data_dir_path)

    if not period:
        period = 1440 # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。下採樣後將資料降為成每分鐘一個數據點，以 1 天 = 1440 分鐘進行觀察。
    print(f'時間步數（time steps）: {period}')
    RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period) # 將batch_size設為1，逐條處理測試數據，記錄每筆數據的預測結果。
    prediction = [] # 用於存儲每個模型的預測值

    # 加載模型並生成預測
    for path_model in tqdm(listdir(path.join(write_result_out_dir, 'model'))):
        file_path = path.join(write_result_out_dir, 'model', path_model)
        print(f'Using Model Name: {target} の {path_model} \n {file_path}')
        best_model = load_model(file_path, custom_objects={'rmse': rmse})
        y_test_pred = best_model.predict_generator(RPG) # 獲取模型對測試數據的預測值。
        prediction.append(y_test_pred)

    # 計算預測結果的準確度，並保存所有的預測數據。集成所有預測結果。
    prediction = np.array(prediction) # 轉換為NumPy陣列，形狀為 (模型數量, 測試數據大小, 預測特徵數=1)。 # 單變量預測結果
    print(f'prediction.shape: {prediction.shape}') # (3, 31942, 1)
    size_test = prediction.shape[1] # 測試數據的樣本數量。
    y_test = y_test[-size_test:] # 將y_test修正為與size_test長度相同，確保測試標籤與預測數據對齊。
    mean_pred = prediction.squeeze() # # 去掉最後一個維度。EX.將數據形狀從 (3, 31942, 1) 轉為 (3, 31942)
    mean_pred = np.mean(mean_pred, axis=0) # 對每行數據取平均值作為集成預測結果，即集成預測值。
    print(f'y_test.shape: {y_test.shape}, pred.shape: {mean_pred.shape}') # 與y_test的形狀一致
        
    # 將預測輸出彙總成CSV保存
    combined_df = pd.DataFrame() # 組合所有預測結果為 DataFrame
    for i in range(prediction.shape[0]):
        combined_df[f'Prediction_{i+1}'] = prediction[i].flatten() # 將每一組預測結果展平為一列
    combined_df['Average'] = mean_pred # 預測平均值
    combined_df.to_csv( path.join(write_result_out_dir, 'combined_predictions.csv') ) # 保存到同一個CSV文件            
    np.save( path.join(write_result_out_dir, target), prediction) # 將所有的預測結果保存為NumPy檔案(.npy)，可用np.load讀取資料。

    # 計算預測誤差    
    mse_score, rmse_loss, mae_loss, mape_loss, msle_loss, r2 = save_mse(y_test, mean_pred, write_result_out_dir) # 計算y_test和mean_pred之間的均方誤差（MSE）分數，同時將模型摘要資訊寫入文件。
    # 繪製圖表
    plt.rcParams['font.size'] = 25 # 設定字體大小
    plt.figure(figsize=(15, 7)) # 建立圖表
    save_prediction_plot(y_test, mean_pred, write_result_out_dir) # 繪製y_test與mean_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
    save_yy_plot(y_test, mean_pred, write_result_out_dir) # 繪製y_test與mean_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
    ResidualPlot(y_test, mean_pred, write_result_out_dir)
    ErrorHistogram(y_test, mean_pred, write_result_out_dir)


def start_ensemble (period, write_out_dir):
    folders = [f for f in listdir(write_out_dir) if path.isdir(path.join(write_out_dir, f))]
    for target in folders:
        ensemble (period, write_out_dir, target)
        keras.backend.clear_session() # 清理記憶體，防止內存堆積。
        print('\n' * 2 + '-' * 140 + '\n' * 2)
    print('おしまい')