from os import makedirs, path, listdir
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# TODO: 觀察MSE預測誤差
def MSE_Improvement(out_dir, train_mode): 
    '''
    比較針對不同條件使用和不使用遷移學習訓練的模型的均方誤差（MSE）值、顯示不同數據集上的遷移學習表現差異。
    '''

    # make output base directory (建立輸出目錄)
    base_out_dir = path.join(out_dir, train_mode)
    makedirs(base_out_dir, exist_ok=True)

    source_relative_path = path.join(out_dir, 'pre-train')
    source_dir = [f for f in listdir(source_relative_path) 
                  if path.isdir(path.join(source_relative_path, f)) and
                  path.exists(path.join(source_relative_path, f, "log.txt"))]
    target_relative_path = path.join(out_dir, 'transfer-learning (Unfreeze)') # TODO: 需要修改指定路徑!
    target_dir = [f for f in listdir(target_relative_path) 
                  if path.isdir(path.join(target_relative_path, f)) and
                  all(path.isfile(path.join(target_relative_path, f, source, "log.txt")) for source in source_dir)]
    
    no_tl, tl = [], []
    for target in target_dir:
        
        # fetch results without transfer learning (收集未使用遷移學習訓練的模型的MSE值。)
        log_path = path.join(out_dir, 'without-transfer-learning', target, 'log.txt')
        with open(log_path, 'r', encoding='utf-8') as f: # 打開目錄中未使用遷移學習訓練的模型的 log.txt 檔。
            lines = f.readlines()
            for line in lines: # 遍歷所有行並提取指標
                if re.match(r"^MAE預測誤差值\s*:", line):
                    base_mae = float(line.split(':')[1].strip()) # 提取MAE值，並將其轉換為浮點數。
                elif re.match(r"^MSE預測誤差值\s*:", line):
                    base_mse = float(line.split(':')[1].strip()) # 提取MSE值，並將其轉換為浮點數。
                # elif re.match(r"^RMSE預測誤差值\s*:", line):
                #     base_rmse = float(line.split(':')[1].strip()) # 提取RMSE值，並將其轉換為浮點數。
                # elif re.match(r"^MAPE預測誤差值\s*:", line):
                #     base_mape = float(line.split(':')[1].strip()) # 提取MAPE值，並將其轉換為浮點數。
                # elif re.match(r"^MSLE預測誤差值\s*:", line):
                #     base_msle = float(line.split(':')[1].strip()) # 提取MSLE值，並將其轉換為浮點數。
                # elif re.match(r"^R2 Score\s*:", line):
                #     r2 = float(line.split(':')[1].strip()) # 提取R2 Score值，並將其轉換為浮點數。                    
            no_tl.append(base_mse)
        print(f'{target}: (MSE: {base_mse})')
        
        # fetch results as row(1×sources) with transfer learning (獲取遷移學習的MSE值)
        row = []
        for source in source_dir:
            log_path = path.join(target_relative_path, target, source, 'log.txt')
            with open(log_path, 'r', encoding='utf-8') as f: # 打開每個相應的log.txt檔。
                lines = f.readlines()
                for line in lines: # 遍歷所有行並提取指標
                    if re.match(r"^MAE預測誤差值\s*:", line):
                        mae = float(line.split(':')[1].strip()) # 提取MAE值，並將其轉換為浮點數。
                    elif re.match(r"^MSE預測誤差值\s*:", line):
                        mse = float(line.split(':')[1].strip()) # 提取MSE值，並將其轉換為浮點數。
                    # elif re.match(r"^RMSE預測誤差值\s*:", line):
                    #     rmse = float(line.split(':')[1].strip()) # 提取RMSE值，並將其轉換為浮點數。
                    # elif re.match(r"^MAPE預測誤差值\s*:", line):
                    #     mape = float(line.split(':')[1].strip()) # 提取MAPE值，並將其轉換為浮點數。
                    # elif re.match(r"^MSLE預測誤差值\s*:", line):
                    #     msle = float(line.split(':')[1].strip()) # 提取MSLE值，並將其轉換為浮點數。
                    # elif re.match(r"^R2 Score\s*:", line):
                    #     r2 = float(line.split(':')[1].strip()) # 提取R2 Score值，並將其轉換為浮點數。
                print('{}:{:.1f} ({})'.format(source, (1 - mse / base_mse) * 100, mse)) # 計算相對改進(MSE改進百分比)：（1 - mse / base_mse） * 100 （百分比）。
                row.append(mse)
        print()
        row.append(base_mse)
        tl.append(row)
    print('※ MSE value in () \n')
    
    # 將結果保存為 DataFrame
    tl = pd.DataFrame(np.array(tl), columns=source_dir+['base'], index=target_dir)
    tl.to_csv(path.join(base_out_dir, 'MSE.csv'), index=True)

    # 計算轉移學習（Transfer Learning）的改進率，並可視化結果。
    metrics_map = (1 - tl.divide(no_tl, axis=0)) * 100 # 計算MSE的改進比例（減少了多少比例的 MSE）。
    metrics_map.to_csv(path.join(base_out_dir, 'MSE Improvement.csv'), index=True) # 保存結果
    
    # 繪製熱圖
    MSE_heatmap = metrics_map.T
    MSE_heatmap.columns = ["MSE Improvement (%)"]
    MSE_heatmap = MSE_heatmap.drop("base") # 移除base行
    MSE_heatmap = MSE_heatmap.sort_values(by="MSE Improvement (%)", ascending=False) # 並按"MSE Improvement (%)"進行排序（降序）
    MSE_heatmap.index = MSE_heatmap.index.str.replace("FishAquaponics_", "", regex=False) # 移除 "FishAquaponics" 前綴
    sns.set(font_scale=2.0) # 設定字體大小
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(MSE_heatmap, annot=True, fmt="1.2f", cmap="coolwarm", linewidths=1,
                cbar_kws={'label': 'mse decreasing rate after transfer [%]'}, 
                annot_kws={"size": 20, "va": "center", "ha": "center"}) # 繪製熱圖（Heatmap），來源模型變為橫軸，目標模型變為縱軸。
    plt.xticks(rotation=0, ha="center", va="center", fontsize=18) # 旋轉橫軸標籤20度。
    plt.yticks(rotation=0, fontsize=18) # 旋轉縱軸標籤20度。
    plt.title("MSE Decreasing Rate After Transfer Learning", fontsize=20, pad=20) # 設置標題
    # 調整色階條標籤
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('MSE Decreasing Rate After Transfer [%]', fontsize=16, labelpad=10)    
    ax.set_ylim(MSE_heatmap.shape[0], 0) # 調整邊界，防止裁切
    plt.tight_layout() # 自動調整圖形中子圖或元素之間的間距，以防止重疊。
    ax.set_ylim(MSE_heatmap.shape[0], 0) # 調整縱軸範圍，避免圖形被裁剪。
    plt.savefig(path.join(base_out_dir, 'MSE Improvement.png')) # 保存圖像
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(base_out_dir, 'MSE Improvement.png')}")
    
    # 根據改進率排序並保存排名
    IoTpond1 = sorted(metrics_map.iloc[0, :].to_dict().items(), key=lambda x: x[1], reverse=True) # 結果是一個按改進率降序排列的 (source_model, improvement) 的列表。
    IoTpond1 = np.array([data[0] for data in IoTpond1]).reshape(-1, 1) # 將排序結果轉換為一列的 2D 陣列。
    rank = pd.DataFrame(np.concatenate([IoTpond1],axis=1), columns=['IoTpond1'])
    rank.to_csv(path.join(base_out_dir, 'MSE Rank.csv'), index=False)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: 觀察MAE預測誤差
def MAE_Improvement(out_dir, train_mode): 
    '''
    比較針對不同條件使用和不使用遷移學習訓練的模型的MAE誤差、顯示不同數據集上的遷移學習表現差異。
    '''

    # make output base directory (建立輸出目錄)
    base_out_dir = path.join(out_dir, train_mode)
    source_relative_path = path.join(out_dir, 'pre-train')
    source_dir = [f for f in listdir(source_relative_path) 
                  if path.isdir(path.join(source_relative_path, f)) and
                  path.exists(path.join(source_relative_path, f, "log.txt"))]
    target_relative_path = path.join(out_dir, 'transfer-learning (Unfreeze)') # TODO: 需要修改指定路徑!
    target_dir = [f for f in listdir(target_relative_path) 
                  if path.isdir(path.join(target_relative_path, f)) and
                   all(path.isfile(path.join(target_relative_path, f, source, "log.txt")) for source in source_dir)]
    
    no_tl, tl = [], []
    for target in target_dir:
        
        # fetch results without transfer learning (收集未使用遷移學習訓練的模型的MSE值。)
        log_path = path.join(out_dir, 'without-transfer-learning', target, 'log.txt')
        with open(log_path, 'r', encoding='utf-8') as f: # 打開目錄中未使用遷移學習訓練的模型的 log.txt 檔。
            lines = f.readlines()
            for line in lines: # 遍歷所有行並提取指標
                if re.match(r"^MAE預測誤差值\s*:", line):
                    base_mae = float(line.split(':')[1].strip()) # 提取MAE值，並將其轉換為浮點數。                  
            no_tl.append(base_mae)
        print(f'{target}: (MAE: {base_mae})')
        
        # fetch results as row(1×sources) with transfer learning (獲取遷移學習的MSE值)
        row = []
        for source in source_dir:
            log_path = path.join(target_relative_path, target, source, 'log.txt')
            with open(log_path, 'r', encoding='utf-8') as f: # 打開每個相應的log.txt檔。
                lines = f.readlines()
                for line in lines: # 遍歷所有行並提取指標
                    if re.match(r"^MAE預測誤差值\s*:", line):
                        mae = float(line.split(':')[1].strip()) # 提取MAE值，並將其轉換為浮點數。
                print('{}:{:.1f} ({})'.format(source, (1 - mae / base_mae) * 100, mae)) # 計算相對改進(MSE改進百分比)：（1 - mse / base_mse） * 100 （百分比）。
                row.append(mae)
        print()
        row.append(base_mae)
        tl.append(row)
    print('※ MAE value in () \n')
    
    # 將結果保存為 DataFrame
    tl = pd.DataFrame(np.array(tl), columns=source_dir+['base'], index=target_dir)
    tl.to_csv(path.join(base_out_dir, 'MAE.csv'), index=True)

    # 計算轉移學習（Transfer Learning）的改進率，並可視化結果。
    metrics_map = (1 - tl.divide(no_tl, axis=0)) * 100 # 計算MSE的改進比例（減少了多少比例的 MSE）。
    metrics_map.to_csv(path.join(base_out_dir, 'MAE Improvement.csv'), index=True) # 保存結果
    
    # 繪製熱圖
    MAE_heatmap = metrics_map.T
    MAE_heatmap.columns = ["MAE Improvement (%)"]
    MAE_heatmap = MAE_heatmap.drop("base") # 移除base行
    MAE_heatmap = MAE_heatmap.sort_values(by="MAE Improvement (%)", ascending=False) # 並按"MSE Improvement (%)"進行排序（降序）
    MAE_heatmap.index = MAE_heatmap.index.str.replace("FishAquaponics_", "", regex=False) # 移除 "FishAquaponics" 前綴
    sns.set(font_scale=2.0) # 設定字體大小
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(MAE_heatmap, annot=True, fmt="1.2f", cmap="seismic", linewidths=1,
                cbar_kws={'label': 'MAE decreasing rate after transfer [%]'}, 
                annot_kws={"size": 20, "va": "center", "ha": "center"}) # 繪製熱圖（Heatmap），來源模型變為橫軸，目標模型變為縱軸。
    plt.xticks(rotation=0, ha="center", va="center", fontsize=18) # 旋轉橫軸標籤20度。
    plt.yticks(rotation=0, fontsize=18) # 旋轉縱軸標籤20度。
    plt.title("MAE Decreasing Rate After Transfer Learning", fontsize=20, pad=20) # 設置標題
    # 調整色階條標籤
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('MAE Decreasing Rate After Transfer [%]', fontsize=16, labelpad=10)    
    ax.set_ylim(MAE_heatmap.shape[0], 0) # 調整邊界，防止裁切
    plt.tight_layout() # 自動調整圖形中子圖或元素之間的間距，以防止重疊。
    ax.set_ylim(MAE_heatmap.shape[0], 0) # 調整縱軸範圍，避免圖形被裁剪。
    plt.savefig(path.join(base_out_dir, 'MAE Improvement.png')) # 保存圖像
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(base_out_dir, 'MAE Improvement.png')}")
    
    # 根據改進率排序並保存排名
    IoTpond1 = sorted(metrics_map.iloc[0, :].to_dict().items(), key=lambda x: x[1], reverse=True) # 結果是一個按改進率降序排列的 (source_model, improvement) 的列表。
    IoTpond1 = np.array([data[0] for data in IoTpond1]).reshape(-1, 1) # 將排序結果轉換為一列的 2D 陣列。
    rank = pd.DataFrame(np.concatenate([IoTpond1],axis=1), columns=['IoTpond1'])
    rank.to_csv(path.join(base_out_dir, 'MAE Rank.csv'), index=False)

'''
mse.csv：記錄了每個來源模型在不同目標模型上的 MSE 值以及基線模型的 MSE。
improvement.csv：包含每個來源模型對不同目標模型的MSE改進率（百分比）。
metrics.png：熱圖，可視化改進率。
rank.csv：列出來源模型的改進率排名。
'''
