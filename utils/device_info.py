from tensorflow.python.client import device_lib #  用來列舉當前系統中所有可用設備的功能，例如 CPU 和 GPU。
import os

with open('utils/device_info.txt', 'w') as f:
    f.write('\n')
    f.write(str(device_lib.list_local_devices())) # 獲取當前系統的所有可用設備（例如，CPU 和 GPU）的詳細資訊。
    f.write('\n')

os.remove('utils/device_info.txt')
