import tensorflow as tf

def limit_gpu_memory():
    '''
    限制TensorFlow使用GPU記憶體的方式，避免一次性佔用所有可用的 GPU 記憶體。
    並啟用在 OOM 時顯示分配的張量資訊。
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')  # 列出可用的 GPU
    if gpus:
        try:
            for gpu in gpus:
                # 設定 GPU 記憶體成長模式，避免一次性佔用過多記憶體
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"成功設定 GPU 記憶體增長模式: {gpu}")
            
            # 設定 OOM 時顯示記憶體分配資訊（TensorFlow 2.x 沒有 RunOptions，因此無法直接設定）
            tf.debugging.set_log_device_placement(True)
        except RuntimeError as e:
            print(f"設定 GPU 記憶體限制時發生錯誤: {e}")

