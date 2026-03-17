import pandas as pd
from tqdm import tqdm


def load_csv(h_file_path, p_file_path,  h_file_list, p_file_list):
    h_df = pd.read_csv(h_file_list).iloc[:, 0]
    p_df = pd.read_csv(p_file_list).iloc[:, 0]

    h_list = h_df.tolist()
    p_list = p_df.tolist()

    h_file = []
    p_file = []
    h_window_per_person = []
    p_window_per_person = []

    for i in tqdm(range(len(h_list)), desc="loading healthy files"):
        data = pd.read_csv(h_file_path + h_list[i], header=None)
        num_row = data.shape[0]

        h_file.append(data)
        h_window_per_person.append(num_row)

    for i in tqdm(range(len(p_list)), desc="loading patient files"):
        data = pd.read_csv(p_file_path + p_list[i], header=None)
        num_row = data.shape[0]

        p_file.append(data)
        p_window_per_person.append(num_row)

    file = h_file + p_file
    window_per_person = h_window_per_person + p_window_per_person

    merged_data = pd.concat(file, ignore_index=True)
    merged_data  = merged_data.to_numpy()

    x = merged_data[:, 1:]
    y = merged_data[:, 0]

    x = x.reshape((x.shape[0], x.shape[1], 1))  # 調整矩陣形狀 (樣本數(window數量), 特徵數(訊號), 1)

    return x, y, window_per_person

