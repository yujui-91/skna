import pandas as pd
import numpy as np
from tqdm import tqdm
#分清健康組/病患組批次讀取並整合 window區間依人

def load_npy(h_file_path, p_file_path,  h_file_list, p_file_list):
    h_list = h_file_list.tolist()
    p_list = p_file_list.tolist()

    h_file = []
    p_file = []
    h_window_per_person = []
    p_window_per_person = []

    for i in tqdm(range(len(h_list)), desc="loading healthy files"):
        data = np.load(h_file_path + f'{h_list[i]}.npy')
        num_row = data.shape[0]

        h_file.append(data)
        h_window_per_person.append(num_row) #分清每個人區間

    for i in tqdm(range(len(p_list)), desc="loading patient files"):
        data = np.load(p_file_path + f'{p_list[i]}.npy')
        num_row = data.shape[0]

        p_file.append(data)
        p_window_per_person.append(num_row)

    file = h_file + p_file
    window_per_person = h_window_per_person + p_window_per_person

    merged_data = np.vstack(file)

    x = merged_data[:, 1:] #singal
    y = merged_data[:, 0] #label 0/1

    x = x.reshape((x.shape[0], x.shape[1], 1))  # 調整矩陣形狀 (樣本數(window數量), 特徵數(訊號), 1)
    #Reshape成3維通常為了餵進 CNN (卷積神經網路) 或 RNN (LSTM/GRU)。
    return x, y, window_per_person

