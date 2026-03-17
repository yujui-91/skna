import pandas as pd
import numpy as np
from tqdm import tqdm

def load_npy(h_file_path, p_file_path, h_file_list, p_file_list):
    print("\n" + "="*50)
    print(">>> 成功！正在執行 data_loader_yujui_1 (Yujui的版本) <<<")
    print("="*50 + "\n")
    # -------------------
    h_list = h_file_list.tolist()
    p_list = p_file_list.tolist()

    h_file = []
    p_file = []
    h_window_per_person = [] 
    p_window_per_person = []

    # 1. 讀取健康組 (Healthy)
    for i in tqdm(range(len(h_list)), desc="loading healthy files"):
        data = np.load(h_file_path + f'{h_list[i]}.npy')
        num_row = data.shape[0]

        h_file.append(data)
        h_window_per_person.append(num_row) # 紀錄健康組這個人的 window 數

    # 2. 讀取病患組 (Patient)
    for i in tqdm(range(len(p_list)), desc="loading patient files"):
        data = np.load(p_file_path + f'{p_list[i]}.npy')
        num_row = data.shape[0]

        p_file.append(data)
        p_window_per_person.append(num_row) # 紀錄病患組這個人的 window 數

    # --- 處理健康組資料 ---
    if len(h_file) > 0:
        h_data_concat = np.vstack(h_file)
        x_h = h_data_concat[:, 1:] 
        y_h = h_data_concat[:, 0]
        # 配合 MultiRocket 格式 (N, T)
        x_h = x_h.reshape((x_h.shape[0], x_h.shape[1])) 
    else:
        x_h, y_h = np.array([]), np.array([])

    # --- 處理病患組資料 ---
    if len(p_file) > 0:
        p_data_concat = np.vstack(p_file)
        x_p = p_data_concat[:, 1:] 
        y_p = p_data_concat[:, 0]
        # 配合 MultiRocket 格式 (N, T)
        x_p = x_p.reshape((x_p.shape[0], x_p.shape[1]))
    else:
        x_p, y_p = np.array([]), np.array([])

    # 回傳順序：
    # 1. 健康特徵, 2. 健康標籤, 3. 健康window數 (List)
    # 4. 病患特徵, 5. 病患標籤, 6. 病患window數 (List)
    return x_h, y_h, h_window_per_person, x_p, y_p, p_window_per_person