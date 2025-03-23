import tensorflow as tf
from keras import backend as K


def limit_gpu_memory():
    '''
    限制TensorFlow使用GPU記憶體的方式，避免一次性佔用所有可用的 GPU 記憶體。
    並啟用在 OOM 時顯示分配的張量資訊。
    '''
