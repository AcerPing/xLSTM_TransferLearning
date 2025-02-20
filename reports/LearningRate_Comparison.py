import os
import re
import pandas as pd


# 定義根目錄 (請替換成你的路徑)
root_dir = r"D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量\Code程式碼\Fwusow_LSTM_TransferLearning\reports\IoTpond_Result"
output_dir = r".\reports\IoTpond_Result\comparison"
os.makedirs(output_dir, exist_ok=True) # 確保輸出目錄存在

# 定義提取指標的函數
def extract_metrics_from_log(file_path):
    
    '''
    比較針對不同初始學習率(Learning Rate)的預測誤差值及S2 Score。
    '''
    
    metrics = {"MAE": None, "MSE": None, "RMSE": None, "MAPE": None, "MSLE": None, "R2": None}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if re.match(r"^MAE預測誤差值\s*:", line):
                metrics["MAE"] = float(line.split(":")[-1].strip())
            elif re.match(r"^MSE預測誤差值\s*:", line):
                metrics["MSE"] = float(line.split(":")[-1].strip())
            elif re.match(r"^RMSE預測誤差值\s*:", line):
                metrics["RMSE"] = float(line.split(":")[-1].strip())
            elif re.match(r"^MAPE預測誤差值\s*:", line):
                metrics["MAPE"] = float(line.split(":")[-1].strip())
            elif re.match(r"^MSLE預測誤差值\s*:", line):
                metrics["MSLE"] = float(line.split(":")[-1].strip())
            elif re.match(r"^R2 Score\s*:", line):
                metrics["R2"] = float(line.split(":")[-1].strip())
    return metrics

# 遍歷所有學習率路徑
results = []
for lr_folder in ['transfer-learning (Unfreeze)_InitLR=1e-4', 'transfer-learning (Unfreeze)_InitLR=1e-5', 'transfer-learning (Unfreeze)_InitLR=1e-6']:
    lr_path = os.path.join(root_dir, lr_folder, 'FishAquaponics_IoTpond1')
    for pond_folder in os.listdir(lr_path):
        pond_path = os.path.join(lr_path, pond_folder)
        log_file = os.path.join(pond_path, "log.txt")
        if os.path.isfile(log_file):
            metrics = extract_metrics_from_log(log_file)
            metrics["Learning Rate"] = lr_folder.split('=')[-1]  # 提取學習率數值
            metrics["Pond"] = pond_folder
            results.append(metrics)

# 轉換為 DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=["Pond", "Learning Rate"]) # 排序並重新排列欄位
column_order = ["Pond", "Learning Rate", "MAE", "MSE", "RMSE", "MAPE", "MSLE", "R2"]
df_results = df_results[column_order]

# 儲存結果到 Excel 檔案
output_file = os.path.join(output_dir, "learning_rate_metrics_comparison.xlsx")
df_results.to_excel(output_file, index=False)
print(f"Results saved to '{output_file}'")
print(df_results)
