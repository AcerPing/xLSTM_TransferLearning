from os import path
from copy import deepcopy
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

from torchinfo import summary # 建議 先安裝torchinfo後，再安裝torch。

# matplotlib 字體設定
plt.rcParams["font.size"] = 13
# 設定中文字體，例如 Noto Sans CJK 字體為默認字體
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
chinese_font = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = [chinese_font.get_name()] + matplotlib.rcParams['font.family']
matplotlib.rcParams['font.sans-serif'] = [chinese_font.get_name()] + matplotlib.rcParams['font.sans-serif'] # 其他備選字體
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題


def save_lr_curve(train_loss, val_loss, out_dir: str, f_name=None):
    """
    save learning curve in deep learning
    Args:
        model : trained model (keras)
        out_dir (str): directory path for saving
    """
    if not isinstance(train_loss, list) or not isinstance(val_loss, list): raise TypeError("❌ train_loss 和 val_loss 必須是 list 類型！")
    f_name = 'Learning Curve' if not f_name else f'{f_name} Learning Curve' # 檔名
    plt.figure(figsize=(30, 10)) # 建立圖表
    plt.rcParams["font.size"] = 18 # 字體大小為 18。
    plt.plot(train_loss, label='Train Loss', marker='o', markersize=5) # 繪製訓練損失曲線
    plt.plot(val_loss, label='Validation Loss', marker='s', markersize=5) # 繪製驗證損失曲線
    # Add value annotations
    for i in range(0, len(train_loss), max(1, len(train_loss)//10)):  # 每 10 個數據點標一次
        plt.annotate(f'{train_loss[i]:.4f}', xy=(i, train_loss[i]), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=12, color='blue', alpha=0.9)
    # Add value annotations for validation loss
    for i in range(0, len(val_loss), max(1, len(val_loss)//10)):  # 每 10 個數據點標一次
        plt.annotate(f'{val_loss[i]:.4f}', xy=(i, val_loss[i]), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=12, color='orange', alpha=0.9)
    plt.title(f'{f_name} (Model Loss)', fontsize=18)
    plt.ylabel('MSE Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 'Validation'], loc='best', fontsize=14)
    plt.grid(alpha=0.3)  # 加入透明網格，便於觀察
    plt.tight_layout()
    plt.savefig(path.join(out_dir, f'{f_name}.png'), bbox_inches='tight')
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, f'{f_name}.png')}")


def save_prediction_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save prediction plot for tareget varibale

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
    """
    plt.figure(figsize=(30, 10)) # 設定圖表大小
    plt.rcParams["font.size"] = 18 # 設置字體大小
    plt.plot([i for i in range(1, 1 + len(y_pred_test_time))], y_pred_test_time, color='red', label='predicted', marker='x', markersize=2) # 繪製預測數據的紅色折線圖
    plt.plot([i for i in range(1, 1 + len(y_test_time))], y_test_time, color='blue', label='measured', marker="o", markersize=2) # 繪製實際數據的藍色折線圖
    
    # 在每個點上顯示數據標籤 (實際數據)
    for i, value in enumerate(y_pred_test_time.flatten()):
        if i % 1000 == 0:  # 每隔 1000 個數據點顯示一次標籤
            plt.annotate(f'{value:.2f}', xy=(i, value), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', color='crimson', fontsize=12, alpha=0.9)
    # 在每個點上顯示數據標籤 (預測數據)
    for i, value in enumerate(y_test_time):
        if i % 1000 == 0:  # 每隔 1000 個數據點顯示一次標籤
            plt.annotate(f'{value:.2f}', xy=(i, value),  xytext=(0, -5), textcoords="offset points", ha='center', va='top', color='dodgerblue', fontsize=12, alpha=0.9)
    
    plt.ylim(0, 1) # 設置y軸的顯示範圍為0到1。
    plt.xlim(0, len(y_test_time)) # 設置x軸範圍，從0到實際數據的長度。
    title = "Comparison of Actual and Predicted Values"
    plt.title(title)
    plt.ylabel('Value') # 設置y軸標籤。
    plt.xlabel('樣本序列') # 設置x軸標籤。
    plt.legend(loc="best") # 顯示圖例，並將圖例放在最佳位置（由 Matplotlib 自動確定）。
    plt.grid(alpha=0.3)  # 加入透明網格，便於觀察
    plt.tight_layout()
    plt.savefig(path.join(out_dir, 'prediction.png'), bbox_inches='tight') # 保存圖像
    # plt.show() # 顯示圖表
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, 'prediction.png')}")


def save_yy_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save yy plot for target variable

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
    """
    plt.figure(figsize=(10, 10)) # 設定圖表大小
    plt.rcParams["font.size"] = 18 # 設置字體大小
    plt.plot(y_test_time, y_pred_test_time, 'b.', label='Predicted vs Observed') # 繪製預測值與實際值的散點圖，使用藍色點標記。
    diagonal = np.linspace(0, 1, 10000) # 繪製對角線
    plt.plot(diagonal, diagonal, 'r-', linestyle='--', label='Ideal Line') # 繪製一條紅色的對角線，表示理想狀況下的預測值與實際值相等。散點越接近這條線，表示預測越準確。
    # Highlight regions for high and low estimates
    plt.fill_between(diagonal, diagonal, 1, color='peachpuff', alpha=0.2, label='Overestimation Region')  # 淺橙色區域 (color='peachpuff '): 表示模型的高估區域（y_pred_test_time > y_test_time）。
    plt.fill_between(diagonal, 0, diagonal, color='palegreen', alpha=0.2, label='Underestimation Region')  # 淺綠色區域 (color='peachpuff '): 表示模型的低估區域（y_pred_test_time < y_test_time）。
    plt.xlim(0, 1) # 設定x軸的範圍為0到1。
    plt.ylim(0, 1) # 設定y軸的範圍為0到1。
    title = '觀察值與預測值的對角線分析圖'
    plt.title(title, fontsize=16)
    plt.xlabel('Observed', fontsize=14) # 設置x軸標籤。
    plt.ylabel('Predicted', fontsize=14) # 設置y軸標籤。
    plt.legend(loc='best', fontsize=12) # Add legend
    plt.tight_layout()
    plt.savefig(path.join(out_dir, 'yy_plot.png'), bbox_inches='tight') # 保存圖像
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, 'yy_plot.png')}")


# 自訂RMSE函數 (內部函數)
def _rmse(mse_loss): # 因為Keras並未內建RMSE作為指標，需要自行定義一個自訂的RMSE指標函數。
    '''
    RMSE 是 mse 的平方根，更直觀地表示誤差，與實際數據單位一致。
    '''
    rmse_loss = np.sqrt(mse_loss)
    return rmse_loss


def save_mse(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, model=None, sequence_length=1440, input_dim=5):
    """
    save mean squared error for tareget variable
    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
        model : trained model (keras)
    """
    # sklearn
    mse_loss = mse(y_test_time, y_pred_test_time) # 計算均方誤差
    rmse_loss = _rmse(mse_loss)  # RMSE
    mae_loss = mae(y_test_time, y_pred_test_time) # 計算平均絕對誤差
    r2 = r2_score(y_test_time, y_pred_test_time)  # R-squared指標，反映模型解釋目標變數變異程度的能力。

    # 實例化 (Keras/TensorFlow 提供的指標，需要定義對象。)
    mape_loss = np.mean(np.abs((y_test_time - y_pred_test_time) / y_test_time)) * 100 # MAPE
    msle_loss = np.mean(np.square(np.log1p(y_test_time) - np.log1p(y_pred_test_time))) # MSLE

    with open(path.join(out_dir, 'log.txt'), 'w') as f: # 寫入文件
        f.write('MAE預測誤差值 : {:.6f}\n'.format(mae_loss))
        f.write('MSE預測誤差值 : {:.6f}\n'.format(mse_loss))
        f.write('RMSE預測誤差值 : {:.6f}\n'.format(rmse_loss))        
        f.write('MAPE預測誤差值 : {:.6f}\n'.format(mape_loss))
        f.write('MSLE預測誤差值 : {:.6f}\n'.format(msle_loss))
        f.write('R2 Score : {:.6f}\n'.format(r2))
        f.write('=' * 65 + '\n')
        if model: # **改用 PyTorch summary**
            f.write("\n=== Model Summary ===\n")
            model_cpu = deepcopy(model).to("cpu")  # **創建模型的副本並移到 CPU**
            f.write(str(model_cpu) + '\n')  # 直接寫入模型結構
            f.write('\n')

            # ✅ 逐層記錄模型資訊
            # f.write("\n=== Model Layers ===\n")
            # for name, module in model_cpu.named_modules():
            #     f.write(f"{name}: {module}\n")

            summary_str = summary(model_cpu, input_size=(1, sequence_length, input_dim), device='cpu', col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=3, verbose=0)  # 適用於PyTorch，且torchsummary.summary() 這個函數不支援GPU模型。
            # depth=3。depth：限制解析深度，避免展開過多層。
            # verbose=0  防止在終端輸出，專注於寫入文件
            f.write(str(summary_str) + '\n')

    return mse_loss, rmse_loss, mae_loss, mape_loss, msle_loss, r2


# 殘差圖（Residual Plot）
def ResidualPlot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    plt.figure(figsize=(12, 8))
    residuals = y_test_time - y_pred_test_time.flatten() # 計算殘差
    plt.scatter(y_pred_test_time, residuals, color='blue', alpha=0.6, label='Residuals (y_test - y_pred)') # 繪製殘差散點圖
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Ideal Line (Residual = 0)') # 基準線 (Residual = 0)
    # 標示模型低估與高估的區域
    plt.fill_between( x=np.linspace(0, 1, 500), y1=0, y2=1, color='lightgreen', alpha=0.2, label='Underestimation Region (Residual > 0)' ) # 淺綠色 (color='lightgreen'): 模型低估區域（殘差 > 0）。
    plt.fill_between( x=np.linspace(0, 1, 500), y1=-1, y2=0, color='lightsalmon', alpha=0.2, label='Overestimation Region (Residual < 0)' )  # 淺橙色 (color='lightsalmon'): 模型高估區域（殘差 < 0）。
    plt.xlim(0, 1) # 設定X軸，預測值範圍0到1。
    plt.ylim(-1, 1) # 設定Y軸，殘差範圍-1到1。
    title = 'Residual Plot 殘差圖'
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black', fancybox=True) # 增加圖例
    plt.tight_layout()
    plt.savefig(path.join(out_dir, 'Residual Plot.png'), bbox_inches='tight') # 保存圖像
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, 'Residual Plot.png')}")
    

# 誤差直方圖（Error Histogram）
def ErrorHistogram(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    plt.figure(figsize=(12, 8))
    residuals = y_test_time - y_pred_test_time.flatten() # 計算殘差
    plt.hist(residuals, bins='auto', color='deepskyblue', label='Residuals (y_test - y_pred)') # 柱狀圖，並讓Matplotlib自動計算bins區間。
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ideal Line (Residual = 0)') # 基準線，加一條紅色的垂直基準線，位於Residual為0的地方。
    # 填充Overestimation區域（Residual < 0）
    plt.axvspan(-1, 0, color='burlywood', alpha=0.2, label='OverEstimation Region (Residual < 0)')  # 當 Residual < 0 時，表示實際值 < 預測值（高估）。
    # 填充Underestimation區域（Residual > 0）
    plt.axvspan(0, 1, color='darkseagreen', alpha=0.2, label='UnderEstimation Region (Residual > 0)')  # 當 Residual > 0 時，表示實際值 > 預測值（低估）。
    title = 'Error Histogram 誤差直方圖'
    plt.title(title, fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('count', fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black', fancybox=True)
    plt.tight_layout()
    plt.savefig(path.join(out_dir, 'Error Histogram.png'), bbox_inches='tight') # 保存圖像
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, 'Error Histogram.png')}")
