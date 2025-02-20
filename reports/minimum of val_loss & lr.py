import pandas as pd

file_path = r'./reports/result/noise-injection/sru/epoch_log.csv' # Load file # 

epoch_log = pd.read_csv(file_path)


# 1. 检查 epoch 是否为 100
if epoch_log['epoch'].count() != 100:
    raise ValueError(f"Expected maximum epoch to be 100, but got {epoch_log['epoch'].max()}.")

# 2. 找到 lr 和 val_loss 的最小值
min_lr = epoch_log['lr'].min()
min_val_loss = epoch_log['val_loss'].min()
min_val_loss_percentage = f"{min_val_loss * 100:.2f}%" # 转换为百分比格式，保留两位小数

print(f"Minimum Learning Rate (lr): {min_lr}")
print(f"Minimum Validation Loss (val_loss): {min_val_loss_percentage}")