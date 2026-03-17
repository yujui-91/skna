import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob, os, re

def load_train_test_folder(filename, pattern, normalise=True):
    # 获取所有符合条件的文件路径
    """
            Load data from a .tsv file into a Pandas DataFrame.
        """
    data_paths = glob.glob(os.path.join(filename, '*'))  # list of all paths
    selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))  # 過濾檔案 搜尋檔名含有skna(pattern)的檔案 如果檔名沒有skna就會被過濾掉
    input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.tsv')]

    def load_data(paths):
        # 初始化一个空列表来存储每个文件的数据数组
        data_list = []

        # 遍历所有路径
        for filename in paths:
            # 使用np.loadtxt读取每个文件的数据
            # data = np.loadtxt(filename, delimiter='\t')
            df = pd.read_csv(filename, delimiter='\t', header=None)  # 無header
            # 将读取的数据添加到列表中
            data_list.append(df)

        # 使用Pandas的concat函数合并DataFrame
        merged_df = pd.concat(data_list, ignore_index=True)
        # 转换为NumPy数组，如果需要
        # merged_data = np.vstack(merged_df)
        total_data = merged_df.to_numpy()

        return total_data


    merged_train_data = load_data(input_paths)

    def data_load(data):
        Y = data[:, 0]
        X = data[:, 1:]

        scaler = StandardScaler()
        for i in range(len(X)):  # 檢查缺失值 如果有缺失值以隨機0 - 0.001之間的值替代
            for j in range(len(X[i])):
                if np.isnan(X[i, j]):
                    X[i, j] = random.random() / 1000
            # scale it later
            if normalise:
                tmp = scaler.fit_transform(X[i].reshape(-1, 1))  # 1D to 2D. 標準化
                X[i] = tmp[:, 0]  # 2D to 1D
        X = X.reshape((X.shape[0], X.shape[1], 1))  # 調整矩陣形狀 (樣本數(window數量), 特徵數(訊號), 1)
        return X, Y

    X, Y = data_load(merged_train_data)

    return X, Y