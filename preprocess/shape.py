import pickle
from os import listdir, path

for dname in listdir('source'): # 計算source底下訓練資料集的樣本數和特徵數。針對於X_train.pkl及X_test.pkl檔案。
    print(dname)
    n_features = 0
    n_samples = 0
    for fname in listdir(path.join('source',dname)):
        full_path = path.join('source', dname, fname)
        if not path.isfile(full_path):  # **跳過資料夾**
            continue
        if not fname.endswith('.pkl'): # **確保只讀取 .pkl 檔案**
            continue
        with open(full_path,'rb') as f:
            data = pickle.load(f)
        # print(f'{fname.replace("pkl","")} : {data.shape}')
        if fname == "X_train.pkl" or fname == "X_test.pkl":
            n_samples += data.shape[0]
            n_features = data.shape[1]
    print(f'Samples : {n_samples}')
    print(f'Features : {n_features}')
    print('-'*30)

for dname in listdir('target'): # 統計target底下訓練和測試資料的樣本數量和特徵數量。
    print(dname)
    n_features = 0
    n_samples = 0
    for fname in listdir(path.join('target',dname)):
        full_path = path.join('target', dname, fname)
        if not path.isfile(full_path):  # **跳過資料夾**
            continue
        if not fname.endswith('.pkl'): # **確保只讀取 .pkl 檔案**
            continue
        with open(full_path,'rb') as f:
            data = pickle.load(f)
        # print(f'{fname.replace("pkl","")} : {data.shape}')
        if fname == "X_train.pkl" or fname == "X_test.pkl":
            n_samples += data.shape[0]
            n_features = data.shape[1]
    print(f'Total : {n_samples}')
    print(f'Features : {n_features}')
    print('-'*30)