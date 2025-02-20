from os import makedirs, path, listdir
import re
import pandas as pd


def metrics_comparison(out_dir, train_mode): 
    '''
    比較針對不同條件使用和不使用遷移學習訓練的模型的均方誤差（MSE）值、顯示不同數據集上的遷移學習表現差異。
    '''

    # make output base directory (建立輸出目錄)
    base_out_dir = path.join(out_dir, train_mode)
    makedirs(base_out_dir, exist_ok=True)

    source_relative_path = path.join(out_dir, 'pre-train')
    source_dir = [f for f in listdir(source_relative_path) if path.isdir(path.join(source_relative_path, f))]
    target_relative_path = path.join(out_dir, 'transfer-learning (Unfreeze)') # TODO: 需要修改指定路徑!
    target_dir = [f for f in listdir(target_relative_path) if path.isdir(path.join(target_relative_path, f))]
    
    # 初始化結果表
    results = {
        "Dataset": [],
        # MAE
        "Transferred MAE": [],
        "Baseline MAE": [],
        "MAE Improvement": [],
        # MSE
        "Transferred MSE": [],
        "Baseline MSE": [],
        "MSE Improvement": [],
        # RMSE
        "Transferred RMSE": [],
        "Baseline RMSE": [],
        "RMSE Improvement": [],
        # MAPE
        "Transferred MAPE": [],
        "Baseline MAPE": [],
        "MAPE Improvement": [],
        # MSLE
        "Transferred MSLE": [],
        "Baseline MSLE": [],
        "MSLE Improvement": [],
        # R2 Score
        "Transferred R2": [],
        "Baseline R2": [],
        "R2 Improvement": []
    }
    
    for target in target_dir:
        # fetch results without transfer learning (收集未使用遷移學習訓練的模型的metrics預測誤差值。)
        baseline_metrics = {} # 初始化 Baseline 指標
        log_path = path.join(out_dir, 'without-transfer-learning', target, 'log.txt')
        with open(log_path, 'r', encoding='utf-8') as f: # 打開目錄中未使用遷移學習訓練的模型的 log.txt 檔。
            lines = f.readlines()
            for line in lines: # 遍歷所有行並提取指標
                if re.match(r"^MAE預測誤差值\s*:", line):
                    base_mae = float(line.split(':')[1].strip()) # 提取MAE值，並將其轉換為浮點數。
                    baseline_metrics["MAE"] = base_mae
                elif re.match(r"^MSE預測誤差值\s*:", line):
                    base_mse = float(line.split(':')[1].strip()) # 提取MSE值，並將其轉換為浮點數。
                    baseline_metrics["MSE"] = base_mse
                elif re.match(r"^RMSE預測誤差值\s*:", line):
                    base_rmse = float(line.split(':')[1].strip()) # 提取RMSE值，並將其轉換為浮點數。
                    baseline_metrics["RMSE"] = base_rmse
                elif re.match(r"^MAPE預測誤差值\s*:", line):
                    base_mape = float(line.split(':')[1].strip()) # 提取MAPE值，並將其轉換為浮點數。
                    baseline_metrics["MAPE"] = base_mape
                elif re.match(r"^MSLE預測誤差值\s*:", line):
                    base_msle = float(line.split(':')[1].strip()) # 提取MSLE值，並將其轉換為浮點數。
                    baseline_metrics["MSLE"] = base_msle
                elif re.match(r"^R2 Score\s*:", line):
                    r2 = float(line.split(':')[1].strip()) # 提取R2 Score值，並將其轉換為浮點數。   
                    baseline_metrics["R2"] = r2
        print(f'{target}: (MSE: {base_mse}, RMSE: {base_rmse}, MAE: {base_mae}, MAPE: {base_mape}, MALE: {base_msle}, R2 Score: {r2})')
        
        # fetch results as row(1×sources) with transfer learning (獲取遷移學習的metrics預測誤差值)
        for source in source_dir:
            log_path = path.join(target_relative_path, target, source, 'log.txt')
            with open(log_path, 'r', encoding='utf-8') as f: # 打開每個相應的log.txt檔。
                lines = f.readlines()
                transferred_metrics = {}
                for line in lines: # 遍歷所有行並提取指標
                    if re.match(r"^MAE預測誤差值\s*:", line):
                        mae = float(line.split(':')[1].strip()) # 提取MAE值，並將其轉換為浮點數。
                        transferred_metrics["MAE"] = mae
                    elif re.match(r"^MSE預測誤差值\s*:", line):
                        mse = float(line.split(':')[1].strip()) # 提取MSE值，並將其轉換為浮點數。
                        transferred_metrics["MSE"] = mse
                    elif re.match(r"^RMSE預測誤差值\s*:", line):
                        rmse = float(line.split(':')[1].strip()) # 提取RMSE值，並將其轉換為浮點數。
                        transferred_metrics["RMSE"] = rmse
                    elif re.match(r"^MAPE預測誤差值\s*:", line):
                        mape = float(line.split(':')[1].strip()) # 提取MAPE值，並將其轉換為浮點數。
                        transferred_metrics["MAPE"] = mape
                    elif re.match(r"^MSLE預測誤差值\s*:", line):
                        msle = float(line.split(':')[1].strip()) # 提取MSLE值，並將其轉換為浮點數。
                        transferred_metrics["MSLE"] = msle
                    elif re.match(r"^R2 Score\s*:", line):
                        r2 = float(line.split(':')[1].strip()) # 提取R2 Score值，並將其轉換為浮點數。
                        transferred_metrics["R2"] = r2
                print('{}:{:.1f} ({})'.format(source, (1 - mse / base_mse) * 100, mse)) # 計算相對改進(MSE改進百分比)：（1 - mse / base_mse） * 100 （百分比）。
                
            results["Dataset"].append(f"{target} ({source})")
            for metric in ["MAE", "MSE", "RMSE", "MAPE", "MSLE", "R2"]:
                results[f"Transferred {metric}"].append(transferred_metrics.get(metric, None))
                results[f"Baseline {metric}"].append(baseline_metrics.get(metric, None))
                if metric in ["R2"]: # R2 的改進計算方式
                    improvement = transferred_metrics.get(metric, None) - baseline_metrics.get(metric, None)  # 使用 Transferred - Baseline 計算 R2 的改進率。
                else: # 其他指標的改進計算方式
                    improvement = (1 - transferred_metrics.get(metric, None) / baseline_metrics.get(metric, None)) * 100  # 使用 (1 - Transferred / Baseline) * 100 計算 MAE、MSE 和 RMSE 的改進率。
                    improvement = f"{improvement:.2f}%"
                results[f"{metric} Improvement"].append(improvement)
        print()
    
    # 將結果轉為 DataFrame
    df_results = pd.DataFrame(results)
    
    # 將 DataFrame 轉置
    df_results = df_results.set_index("Dataset").transpose() # 將Dataset列設置為索引，這樣在轉置時它會成為表格的欄位名稱。
    
    # 將表格保存為 CSV
    output_path = path.join(base_out_dir, 'Metrics Improvement.csv')
    df_results.to_csv(output_path) # index=False
    
    # 打印完成訊息
    print("表格已整理完成並保存。")
    

# out_dir = r'D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量\Code程式碼\Fwusow_LSTM_TransferLearning\reports\IoTpond_Result'
# train_mode = r'comparison'
# comparison(out_dir, train_mode)
