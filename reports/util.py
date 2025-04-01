from os import listdir, path
import pickle

from dtw import dtw # 計算動態時間規劃距離（Dynamic Time Warping）。
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib.pyplot as plt


def Feature_Unsimilarity(out_dir, train_mode): # feature_make, diff_cal=None
    '''
    比較各數據與目標資料集之間的相似性，
    本程式碼通過 動態時間規劃 和 曼哈頓距離 測量資料序列之間的相似程度。
    結果中數值越小，代表序列越相似；數值越大，代表差異越大。
    '''

    # load dataset (載入數據集)
    source_path = './dataset/source/'
    data_dict = {}
    for d_name in listdir(source_path):
        with open(path.join(source_path, d_name, 'y_test.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_dict[d_name] = data # 數據集以字典形式存儲，鍵為數據集名稱，值為數據。
    
    target_path = './dataset/target/'
    for d_name in listdir(target_path):
        with open(path.join(target_path, d_name, 'y_test.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_dict[d_name] = data # 數據集以字典形式存儲，鍵為數據集名稱，值為數據。

    # calculate difference
    dataset_index = pd.DataFrame(columns=listdir(target_path), index=listdir(source_path))

    # 計算數據集的相似性指數, 比較各數據集與目標資料集的相似性。
    for key, value in data_dict.items():
        target = listdir(target_path)[0]
        if key == target : continue
        print(f'比較{key}的數據相似程度。')
        
        # Dynamic Time Warping 衡量兩個時間序列相似度的演算法，透過非線性地調整時間軸，對齊兩個序列。
        # 距離計算方式，有"歐幾里得距離" or "曼哈頓距離"。 #-- "distance：DTW 距離值" vs. "path：最佳對齊路徑"
        
        # "曼哈頓距離"
        manhattan_distance = lambda x, y: np.abs(x - y) # 設定曼哈頓距離作為DTW的距離測量方法
        dataset_index.at[key, target] = dtw(data_dict[target], data_dict[key], manhattan_distance).distance # 使用曼哈頓距離計算DTW距離
        
        # "歐幾里得距離"
        # euclidean_distance = lambda x, y: np.sqrt((x - y) ** 2)  # 設定歐幾里得距離作為DTW的距離測量方法
        # dataset_index.at[key, target] = dtw(data_dict[target], data_dict[key], euclidean_distance).distance # 使用歐幾里得距離計算DTW距離 
        
    base_out_dir = path.join(out_dir, train_mode)
    print(dataset_index)
    dataset_index.to_csv(path.join(base_out_dir, 'DTW特徵非類似度(曼哈頓距離).csv'), index=True)
    return target_path, source_path, base_out_dir, dataset_index

# ---------------------------------------------------------------------------------------------------------------------------------

def FeatureUnsimilarity_with_MSE (target_path, source_path, base_out_dir, dataset_index):
    
    '''
    特徵非相似程度與MSE改善程度
    '''
    
    # 讀取MSE改善程度檔案
    mse_df = pd.read_csv(path.join(base_out_dir, 'MSE.csv'), index_col=0)
    mse_improvement = pd.read_csv(path.join(base_out_dir, 'MSE Improvement.csv'), index_col=0)
    print('mse_df','\n', mse_df)
    print('mse_improvement','\n', mse_improvement)
    
    # First Image - MSE with Feature Similarity
    # 比較有無遷移學習(転移学習あり 與 転移学習なし)的結果差異。X 軸：特徵非類似度，表示來源與目標資料集的相似性程度；Y 軸：MSE，表示預測效果的準確性。
    plt.figure(figsize=(20, 10))  # # 調整畫布大小以避免擠壓。 整體畫布寬度 12 英吋，高度 6 英吋
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1) # 創建 1 行 2 列的子圖
        x, y = [], []
        for source in listdir(source_path):
            x.append(dataset_index.at[source, target]*10**7) # x：來源資料集與目標資料集的特徵非類似度。特徵非類似度被乘以 10^7 進行放大，以更好地顯示數據。
            y.append(mse_df.at[target, source]) # y：來源資料集與目標資料集的MSE。
            
        # MSE改善(royalblue深藍色)或惡化(#CD5C5C磚紅色)顏色標註
        colors = ['#CD5C5C' if mse_df.at[target, source] > mse_df.at[target, 'base'] else 'royalblue' for source in listdir(source_path)]
        for i, (xi, yi, color) in enumerate(zip(x, y, colors)):
            plt.plot(xi, yi, '.', markersize=25, label='with transfer' if i == 0 else None, color=color) # 表示每個來源資料集的特徵非類似度與MSE，表示使用遷移學習的結果。
            # 辨識每個資料點對應的來源資料集
            plt.annotate(listdir(source_path)[i].replace("FishAquaponics_", ""), 
                     xy=(xi, yi), xytext=(8, 8), textcoords="offset points", fontsize=16, color='black', 
                     arrowprops=dict(arrowstyle='-', color='gray'))        
        
        x_range = max(x) - min(x) # 計算特徵非類似度的範圍，用於繪製基線。
        x_min = min(x) - 0.1 * x_range
        x_max = max(x) + 0.1 * x_range
        plt.plot([x_min, x_max], [mse_df.at[target, 'base'] for _ in range(2)], linestyle='dashed', linewidth=3, label='without transfer', color='red') # 無遷移學習情況下的MSE，繪製為虛線。
        plt.text(x_min, mse_df.at[target, 'base'] - 0.0005, 'without-transfer-learning', fontsize=14, color='black', va='bottom') # 在虛線旁加上標籤
        
        # 填充背景區域顏色
        if any(yi >= mse_df.at[target, 'base'] for yi in y): # 判斷是否有數據點落在 Worse Region
            plt.fill_betweenx(
                y=[mse_df.at[target, 'base'], max(y)],
                x1=x_min,
                x2=x_max,
                color='#FFC0CB',
                alpha=0.2,
                label='Worse Region') # 惡化用lightcoral表示，顯示負面結果。
        
        if any(yi < mse_df.at[target, 'base'] for yi in y): # 判斷是否有數據點落在 Better Region
            plt.fill_betweenx(
                y=[min(y), mse_df.at[target, 'base']],
                x1=x_min,
                x2=x_max,
                color='lightgreen',
                alpha=0.2,
                label='Better Region') # 改善用lightgreen表示，象徵進步。
            
        plt.xlabel('特徵非類似度（數值越大，越不相似。）', fontsize=14) # 設定X軸名稱。
        plt.ylabel('MSE Loss（預測誤差）', fontsize=14) # 設定Y軸名稱。
        plt.title( f'{target}'.replace("FishAquaponics_", "") + '的特徵非類似度與MSE改善率')
        plt.legend(loc='best', fontsize=14) # 顯示標籤資訊。

    plt.tight_layout() # ：自動調整子圖間距，避免重疊。
    plt.grid(alpha=0.3)  # 加入透明網格，便於觀察
    plt.savefig( path.join(base_out_dir, '特徵相似性與MSE的關係圖'), bbox_inches='tight' ) # 保存圖表
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(base_out_dir, '特徵相似性與MSE的關係圖')}")

    # Second Image - plot dataset_idx rank vs improvement (繪製相似性排序與改進程度的關係圖)
    plt.figure(figsize=(20, 10))  # # 調整畫布大小以避免擠壓。 整體畫布寬度 12 英吋，高度 6 英吋
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1)
        sources_sorted = dataset_index[target].sort_values().keys().tolist()  # 獲取排序後的 source 名稱列表
        improvement_list = [mse_improvement.at[target, source] for source in sources_sorted]  
        n = len(improvement_list)
        plt.plot(range(1, n + 1), improvement_list, 'b', label='with transfer')
        plt.plot(range(1, n + 1), [0 for _ in range(len(improvement_list))], 'r', label='without transfer', linestyle='dashed') 
        # 在每個點上標註數據集名稱
        for rank, (improvement, source) in enumerate(zip(improvement_list, sources_sorted), start=1):
            plt.annotate(
                source,  # 要標註的文字
                xy=(rank, improvement),  # 點的座標
                xytext=(5, 5),  # 偏移量
                textcoords='offset points',  # 偏移基於點的座標系
                fontsize=8,  # 字體大小
                color='black'  # 文字顏色
            )
        plt.xlabel('Feature Similarity Rank / -', fontweight='bold')
        plt.ylabel('Improvement / %', fontweight='bold')
        plt.yticks([i*50 for i in range(-3,4)])
        plt.legend(loc='best')
        plt.title(f'({"ab"[idx]}) {target}')
    plt.tight_layout()
    plt.savefig( path.join(base_out_dir, '特徵相似性與MSE改進程度的關係圖'), bbox_inches='tight' ) # 保存圖表
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(base_out_dir, '特徵相似性與MSE改進程度的關係圖')}")
    
# ---------------------------------------------------------------------------------------------------------------------------------

def FeatureUnsimilarity_with_MAE (target_path, source_path, base_out_dir, dataset_index):
    
    '''
    特徵非相似程度與MAE改善程度
    '''
    
    # 讀取MAE改善程度檔案
    mae_df = pd.read_csv(path.join(base_out_dir, 'MAE.csv'), index_col=0)
    print('mae_df','\n', mae_df)
    
    # First Image - MAE with Feature Similarity
    # 比較有無遷移學習(転移学習あり 與 転移学習なし)的結果差異。X 軸：特徵非類似度，表示來源與目標資料集的相似性程度；Y 軸：MSE，表示預測效果的準確性。
    plt.figure(figsize=(20, 10))  # # 調整畫布大小以避免擠壓。 整體畫布寬度 12 英吋，高度 6 英吋
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1) # 創建 1 行 2 列的子圖
        x, y = [], []
        for source in listdir(source_path):
            x.append(dataset_index.at[source, target]*10**7) # x：來源資料集與目標資料集的特徵非類似度。特徵非類似度被乘以 10^7 進行放大，以更好地顯示數據。
            y.append(mae_df.at[target, source]) # y：來源資料集與目標資料集的MSE。
            
        # MSE改善(royalblue深藍色)或惡化(#CD5C5C磚紅色)顏色標註
        colors = ['#CD5C5C' if mae_df.at[target, source] > mae_df.at[target, 'base'] else 'royalblue' for source in listdir(source_path)]
        for i, (xi, yi, color) in enumerate(zip(x, y, colors)):
            plt.plot(xi, yi, '.', markersize=25, label='with transfer' if i == 0 else None, color=color) # 表示每個來源資料集的特徵非類似度與MSE，表示使用遷移學習的結果。
            # 辨識每個資料點對應的來源資料集
            plt.annotate(listdir(source_path)[i].replace("FishAquaponics_", ""), 
                     xy=(xi, yi), xytext=(8, 8), textcoords="offset points", fontsize=16, color='black', 
                     arrowprops=dict(arrowstyle='-', color='gray'))        
        
        x_range = max(x) - min(x) # 計算特徵非類似度的範圍，用於繪製基線。
        x_min = min(x) - 0.1 * x_range
        x_max = max(x) + 0.1 * x_range
        plt.plot([x_min, x_max], [mae_df.at[target, 'base'] for _ in range(2)], linestyle='dashed', linewidth=3, label='without transfer', color='red') # 無遷移學習情況下的MSE，繪製為虛線。
        plt.text(x_min, mae_df.at[target, 'base'] + 0.0005, 'without-transfer-learning', fontsize=14, color='black', va='bottom') # 在虛線旁加上標籤
        
        # 填充背景區域顏色
        if any(yi >= mae_df.at[target, 'base'] for yi in y): # 判斷是否有數據點落在 Worse Region
            plt.fill_betweenx(
                y=[mae_df.at[target, 'base'], max(y)],
                x1=x_min,
                x2=x_max,
                color='#FFC0CB',
                alpha=0.2,
                label='Worse Region') # 惡化用lightcoral表示，顯示負面結果。
        
        if any(yi < mae_df.at[target, 'base'] for yi in y): # 判斷是否有數據點落在 Better Region
            plt.fill_betweenx(
                y=[min(y), mae_df.at[target, 'base']],
                x1=x_min,
                x2=x_max,
                color='lightgreen',
                alpha=0.2,
                label='Better Region') # 改善用lightgreen表示，象徵進步。
            
        plt.xlabel('特徵非類似度（數值越大，越不相似。）', fontsize=14) # 設定X軸名稱。
        plt.ylabel('MAE Loss（預測誤差）', fontsize=14) # 設定Y軸名稱。
        plt.title( f'{target}'.replace("FishAquaponics_", "") + '的特徵非類似度與MAE改善率')
        plt.legend(loc='best', fontsize=14) # 顯示標籤資訊。

    plt.tight_layout() # ：自動調整子圖間距，避免重疊。
    plt.grid(alpha=0.3)  # 加入透明網格，便於觀察
    plt.savefig( path.join(base_out_dir, '特徵相似性與MAE的關係圖'), bbox_inches='tight' ) # 保存圖表
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(base_out_dir, '特徵相似性與MAE的關係圖')}")
    
# ---------------------------------------------------------------------------------------------------------------------------------

def dataset_idx_vs_improvement (out_dir, train_mode, diff_cal=None) :
    '''   
    Parameters:
    - out_dir (str): 輸出目錄
    - train_mode (str): 訓練模式
    '''
    # matplotlib設定，包含中文字體設定 (例如 Noto Sans CJK 字體為默認字體)。
    plt.rcParams["xtick.direction"] = "in" # 控制Matplotlib圖表的x軸刻度線方向。"in"：刻度線指向圖表內部。
    plt.rcParams["ytick.direction"] = "in" # 控制Matplotlib圖表的y軸刻度線方向。"in"：刻度線指向圖表內部。
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    chinese_font = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = [chinese_font.get_name()] + rcParams['font.family']
    rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    rcParams['font.sans-serif'] = [chinese_font.get_name()] + rcParams['font.sans-serif'] # 其他備選字體

    target_path, source_path, base_out_dir, dataset_index = Feature_Unsimilarity (out_dir, train_mode) # 特徵不相似度計算
    FeatureUnsimilarity_with_MSE (target_path, source_path, base_out_dir, dataset_index) # 特徵非相似程度與MSE改善程度
    FeatureUnsimilarity_with_MAE (target_path, source_path, base_out_dir, dataset_index) # 特徵非相似程度與MAE改善程度
    
