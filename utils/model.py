import os
from keras.layers import Input, Dense, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import initializers, regularizers
import keras.backend as K
import numpy as np


# 自訂 RMSE 函數
def rmse(y_true, y_pred): # 因為Keras並未內建RMSE作為指標，需要自行定義一個自訂的RMSE指標函數。
    '''
    RMSE 是 mse 的平方根，更直觀地表示誤差，與實際數據單位一致。
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def build_model(input_shape: tuple, # 模型的輸入形狀(timesteps, features)
                gpu, # 若為True則使用CUDA加速的LSTM層（CuDNNLSTM）
                write_result_out_dir,
                pre_model=None, # 若有傳入預訓練模型，則可以從中載入權重。
                freeze=False, # 若為True，會將部分層設為不可訓練，用於遷移學習。
                noise=None, # 若設定此參數，會加入一層高斯噪聲層，模擬數據變異。
                verbose=True,
                savefig=True):

    if gpu:
        from keras.layers import CuDNNLSTM as LSTM
        print(f'開啟GPU顯卡進行運算: {gpu}')
    else:
        from keras.layers import LSTM

    # construct the model
    input_layer = Input(input_shape)
    
    print(f'是否加入噪聲: {noise}')
    if noise:
        noise_input = GaussianNoise(np.sqrt(noise))(input_layer) # 加入高斯噪聲層，用於模擬數據的隨機變異。屬於正則化技術，而非數據擴充（Data Augmentation）。np.sqrt(noise) 表示噪聲的標準差。
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
                kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性，並且有效減少梯度消失或梯度爆炸問題。
                bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
            )
        )(noise_input)

    else:
        dense = TimeDistributed(
            Dense( 
                10, # 輸出單元數為10的全連接層。
                kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
                kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
                bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
            )
        )(input_layer)

    lstm1 = LSTM(
        60,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        recurrent_initializer=initializers.Orthogonal(seed=0), # 將 LSTM 的遞歸權重初始化為正交矩陣，以促進梯度穩定。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(dense)
    lstm1 = BatchNormalization()(lstm1) # 正規化，穩定訓練過程、加速收斂，並提高模型的泛化能力。

    lstm2 = LSTM(
        60,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        recurrent_initializer=initializers.Orthogonal(seed=0), # 將 LSTM 的遞歸權重初始化為正交矩陣，以促進梯度穩定。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(lstm1)
    lstm2 = BatchNormalization()(lstm2) # 正規化，穩定訓練過程、加速收斂，並提高模型的泛化能力。

    output_layer = Dense(
        1,
        activation='sigmoid', # 激活函數為sigmoid，適合輸出一個範圍在0到1之間的預測結果。
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(lstm2)

    model = Model(inputs=input_layer, outputs=output_layer) # 建立模型
    if savefig:
        plot_model(model, to_file=f'{write_result_out_dir}/architecture.png', show_shapes=True, show_layer_names=True) # 繪製神經網路模型的結構並將其保存為圖片檔案
    

    # transfer weights from pre-trained model (遷移學習：載入預訓練模型權重)
    if pre_model:
        for i in range(2, len(model.layers) - 1): #（跳過輸入層和最後輸出層）           
            print(f"\n--- Layer {i}: {model.layers[i].name} ---")

            # 獲取當前層的原始權重
            original_weights = model.layers[i].get_weights() # 模型當前層的初始權重。
            # print(f"Original Weights (Before):\n{original_weights}")
            
            # 將對應層的權重從預訓練模型載入
            pre_trained_weights = pre_model.layers[i].get_weights() # 從預訓練模型中加載的權重。
            # print(f"Pre-trained Weights (Loaded):\n{pre_trained_weights}")
            
            # 獲取更新後的權重            
            model.layers[i].set_weights(pre_model.layers[i].get_weights()) # 將對應層的權重從預訓練模型載入。

            # 獲取更新後的權重
            updated_weights = model.layers[i].get_weights() # 載入預訓練權重後的權重。
            # print(f"Updated Weights (After):\n{updated_weights}")

            # 檢查是否與預訓練模型一致
            if all(np.array_equal(w1, w2) for w1, w2 in zip(updated_weights, pre_trained_weights)):
                print("Weights successfully updated to pre-trained weights!")
            else:
                print("Weights mismatch after update!")

            if freeze: # 若 freeze=True，則將這些層設置為不可訓練（即權重不會在訓練中更新），這樣可以保持預訓練權重不變。
                model.layers[i].trainable = False
                print(f"Layer {i} ({model.layers[i].name}) is now frozen and will not be updated during training.")
            else: # 否則，權重可訓練。允許模型在新數據上學習特定模式。
                model.layers[i].trainable = True # 保證層被設置為可訓練（防止之前被凍結）
                print(f"Layer {i} ({model.layers[i].name}) is trainable and its weights will be updated during training (fine-tuning).")

    # 調整優化器&學習率。
    if pre_model:
        # 定義每個數據集的學習率。預訓練模型的微調通常需要更小的學習率。
        dataset_learning_rates = {
            'FishAquaponics_IoTpond2': 1e-5,  # 針對 IoTpond2
            'FishAquaponics_IoTpond3': 1e-4,  # 針對 IoTpond3
            'FishAquaponics_IoTpond4': 1e-4,  # 針對 IoTpond4
        }
        current_dataset = write_result_out_dir.split(os.sep)[-1]  # 根據系統路徑分隔符分割路徑，取得最後一個路徑部分。
        init_learning_rate = dataset_learning_rates.get(current_dataset, 1e-4)  # 默認學習率為 1e-4
    else:
        # 其它
        dataset_learning_rates = {
            'FishAquaponics_IoTpond3': 1e-5,  # 針對 IoTpond3
        }
        current_dataset = write_result_out_dir.split(os.sep)[-1]  # 根據系統路徑分隔符分割路徑，取得最後一個路徑部分。
        init_learning_rate = dataset_learning_rates.get(current_dataset, 1e-4)  # 默認學習率為 1e-4，適合大多數模型的初始訓練。
    print(f'初始學習率: {init_learning_rate}')
    Adam_optimizer = Adam(learning_rate=init_learning_rate) # 標準Adam優化器
    print(f'優化器參數: {Adam_optimizer.get_config()}')
    model.compile(optimizer=Adam_optimizer, loss='mse', metrics=['mse', rmse, 'mae','mape','msle']) # metrics是模型訓練過程中用來監控模型性能的指標、評估模型的訓練效果。 # 使用動態學習率調整策略（如 ReduceLROnPlateau），設定初始學習率有助於更好地控制學習率範圍。
    if verbose: print(model.summary())

    return model
    
'''
GaussianNoise 高斯噪聲層
好處：
1.) 防止過擬合：模型會學習到更多數據的變異性，而不是過度擬合訓練數據的特定模式。
2.) 增加泛化能力：模型在訓練時遇到更多不同的數據輸入，從而提高模型在測試數據上的性能。
3.) 模擬數據噪聲：幫助模型在面對真實世界的噪聲數據時表現得更好。

L2 正則化 
目的：減少過度擬合，也就是讓模型不過於貼合訓練數據，使它能對新數據有更好的表現。
原理：L2 正則化會讓模型的權重（即每個輸入對預測結果的影響程度）保持較小，以便更簡單、更平滑地適應數據。
作法：L2 正則化通過在損失函數中加入一項與權重的平方和成比例的懲罰項來抑制模型的複雜度，從而減少過度擬合風險。
如何實現：它會把每個權重值的平方乘上一個小係數（如 0.01），然後把這些結果加到損失函數中。這會讓模型更偏好小權重，從而減少過度擬合的風險。

Orthogonal 初始化
目的：初始化模型權重，使模型在一開始訓練時更穩定。
原理：Orthogonal 初始化會讓模型的初始權重彼此間保持「正交」，也就是說，它們的方向完全不相關。這樣能確保訊號在傳遞過程中不會過於減弱或變得過強，這對於 RNN 等時間序列模型特別有幫助。
如何實現：在模型開始訓練前，為權重分配一組特殊的初始值，使它們的方向是彼此「垂直」的（數學上稱之為「正交矩陣」）。

Glorot 均勻初始化方法 是什麼？
目的：確保模型一開始的權重不會讓數據信號在傳遞過程中爆炸或消失。
原理：Glorot 均勻初始化（也叫 Xavier 初始化）會根據輸入與輸出的神經元數量，選擇一個適當的範圍，從中均勻分布地選取權重值。這樣可以保持訊號穩定，避免模型在訓練初期就出現梯度爆炸或消失的問題。
如何實現：模型的權重會隨機從 [-limit, limit] 的範圍中選取，其中 limit 是基於該層的輸入和輸出單元數計算得來的。

BatchNormalization (批次正規化) 是什麼？
目的：加速訓練，並讓模型更穩定。
原理：批次正規化會在每一層的輸出上進行「標準化」，也就是把輸出調整成一個更穩定的範圍（通常平均值為 0，標準差為 1）。這樣做可以讓模型更快找到最佳解，減少訓練過程中的波動。
如何實現：模型會根據當前的批次數據，計算出每層輸出的平均值和標準差，然後把每個輸出都調整到該範圍內。這樣做可以平衡輸出的大小，提升模型的穩定性和收斂速度。
'''
