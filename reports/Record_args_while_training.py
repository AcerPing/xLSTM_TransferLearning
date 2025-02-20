from os import path
from datetime import datetime

def Record_args_while_training(write_out_dir, train_mode, source, nb_batch, bsize, period, data_size):
    with open(path.join(write_out_dir, train_mode, f'nb_batch{nb_batch} training_parameters.txt'), 'a') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{current_time}]\n")
        f.write(f"訓練資料集 {source} 時的參數：\n")
        f.write(f"Data Size (data_size): {data_size}\n")
        f.write(f"Number of Batches (args['nb_batch']): {nb_batch}\n")
        f.write(f"Batch Size (bsize): {bsize}\n")
        f.write(f"Sequence Length (period): {period}\n")
        
        
