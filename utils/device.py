import tensorflow as tf
from keras import backend as K


def limit_gpu_memory():
    '''
    限制TensorFlow使用GPU記憶體的方式，避免一次性佔用所有可用的 GPU 記憶體。
    並啟用在 OOM 時顯示分配的張量資訊。
    '''
    config = tf.ConfigProto() # 用來設置TensorFlow的運行配置。
    config.gpu_options.allow_growth = True # 允許TensorFlow根據需求動態增加GPU記憶體的使用量
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True) # 啟用OOM錯誤時顯示分配的張量資訊
    sess = tf.Session(config=config) # 建立Session 
    sess._config = run_options  # 在 Session中加入選項
    K.set_session(sess) # # 設定Keras的後端Session
