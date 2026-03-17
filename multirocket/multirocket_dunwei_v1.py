import argparse
import os
import platform
import socket
import time
from datetime import datetime
from joblib import dump  # 儲存模型使用
import numba
import numpy as np
import pandas as pd
import pytz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from multirocket.multirocket import MultiRocket
from utils.data_loader import read_univariate_ucr, non_109_datasets, read_univariate_ucr_skna, read_univariate_ucr_bladder, load_train_test_folder
from utils.data_loader_2 import load_npy
from utils.tools import create_directory
from multirocket.logistic_regression import LogisticRegression
pd.set_option('display.max_columns', 500)


parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--datapath", type=str, required=False, default=r"V:\Oswin\MACE\dataset\SKNA_signal\window\ch1/")  # 學長檔案路徑
parser.add_argument("-d", "--datapath", type=str, required=False, default=r"F:\M143020071\raw_data_result\SKNA_signal\ch1/") 
parser.add_argument("-p", "--problem", type=str, required=False, default="sr2500_500_1000_20s_ECG_signal_rpeak_5_10min_longer_100pts_mr_20win") 
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000) # 不能更改, original 50000
parser.add_argument("-t", "--num_threads", type=int, required=False, default=10)  # -1表示使用全部線程
parser.add_argument("-s", "--save", type=bool, required=False, default=True)
parser.add_argument("-v", "--verbose", type=int, required=False, default=2)
parser.add_argument("--patience", type=int, required=False, default=20, help='總訓練損失連續10次epoch保持一致的損失值或沒有再更佳的表現(損失沒下降)')  # 沒用到
parser.add_argument("--patience_lr", type=int, required=False, default=5, help='訓練學習速率連續5次保持一致的損失值或沒有再更佳的表現(損失沒下降)')  # 沒用到

arguments = parser.parse_args()

if __name__ == '__main__':
    data_path = arguments.datapath
    problem = arguments.problem
    num_features = arguments.num_features
    num_threads = arguments.num_threads
    itr = arguments.iter
    save = arguments.save
    verbose = arguments.verbose

    # output_path = os.getcwd() + "/output/"
    output_path = r"F:\M143020071\multirocket_result" 
    classifier_name = "MultiRocket_{}".format(num_features)
    data_folder = data_path + problem + "/"
    x_transform_path = data_folder + "/x_transform/"

    if not os.path.exists(x_transform_path):
        os.makedirs(x_transform_path)
        print(f"Path created: {x_transform_path}")
    else:
        print(f"Path already exists: {x_transform_path}")

    if os.path.exists(data_folder):
        if num_threads > 0:
            numba.set_num_threads(num_threads)

        start = time.perf_counter()

        output_dir = "{}/multirocket/resample_{}/{}/{}/".format(
            output_path,
            itr,
            classifier_name,
            problem
        )
        if save:
            create_directory(output_dir)

        print("=======================================================================")
        print("Starting Experiments")
        print("=======================================================================")
        print("Data path: {}".format(data_path))
        print("Output Dir: {}".format(output_dir))
        print("Iteration: {}".format(itr))
        print("Problem: {}".format(problem))
        print("Number of Features: {}".format(num_features))

        #以下需要改 
        if os.path.exists(data_folder + 'X_train.npy'):
            print("Loading data from npy")
            X_train = np.load(data_folder + 'X_train.npy')
            y_train = np.load(data_folder + 'y_train.npy')
            X_test = np.load(data_folder + 'X_test.npy')
            y_test = np.load(data_folder + 'y_test.npy')
        else:
            train_h_df = pd.read_csv(data_folder + "train_h_file_name.csv").iloc[:, 0]
            train_p_df = pd.read_csv(data_folder + "train_p_file_name.csv").iloc[:, 0]
            test_h_df = pd.read_csv(data_folder + "test_h_file_name.csv").iloc[:, 0]
            test_p_df = pd.read_csv(data_folder + "test_p_file_name.csv").iloc[:, 0]

            X_train, y_train, train_window_per_person = load_npy(data_folder + "non_mace_zscore/",
                                                                 data_folder + "mace_zscore/",
                                                                 train_h_df,
                                                                 train_p_df)

            X_test, y_test, test_window_per_person = load_npy(data_folder + "non_mace_zscore/",
                                                              data_folder + "mace_zscore/",
                                                              test_h_df,
                                                              test_p_df)
            
            train_wpp_df = pd.DataFrame(train_window_per_person, columns=["window_per_person"])
            test_wpp_df = pd.DataFrame(test_window_per_person, columns=["window_per_person"])

            train_file_name = pd.concat([train_h_df, train_p_df], axis=0, ignore_index=True)
            test_file_name = pd.concat([test_h_df, test_p_df], axis=0, ignore_index=True)

            train_wpp_df = pd.concat([train_file_name, train_wpp_df], axis=1)
            test_wpp_df = pd.concat([test_file_name, test_wpp_df], axis=1)

            train_wpp_df.to_csv(data_folder + "train_window_per_person.csv", index=False)
            test_wpp_df.to_csv(data_folder + "test_window_per_person.csv", index=False)
        
            np.save(data_folder + 'X_train.npy', X_train)  
            np.save(data_folder + 'y_train.npy', y_train)
            np.save(data_folder + 'X_test.npX_trainy', X_test)
            np.save(data_folder + 'y_test.npy', y_test)
        
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)  ##no train test  要改成只有y
        y_test = encoder.transform(y_test)

        # returns ntc format, remove the last dimension
        X_train = X_train.reshape((.shape[0], X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
        #改到這邊-----------------------------------------------
        #以下不用改
        if (itr > 0) and (problem not in non_109_datasets):  # 不知道這邊在用意
            all_data = np.vstack((X_train, X_test))  # train test
            # all_data = np.vstack(X_train) #train
            all_labels = np.hstack((y_train, y_test))  # train test
            # all_labels = np.hstack(y_train) #train
            print(all_data.shape)

            all_indices = np.arange(len(all_data))
            training_indices = np.loadtxt("data/indices109/{}_INDICES_TRAIN.txt".format(problem),
                                          skiprows=itr,
                                          max_rows=1).astype(np.int32)
            test_indices = np.setdiff1d(all_indices, training_indices, assume_unique=True)

            X_train, y_train = all_data[training_indices, :], all_labels[training_indices]
            X_test, y_test = all_data[test_indices, :], all_labels[test_indices]

        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        # nb_classes = len(np.unique(np.concatenate(y_train, axis=0)))
        classifier = MultiRocket(
            num_features=num_features,
            classifier="logistic",
            verbose=verbose
        )

        # is_debug = False
        # if is_debug:
        #     num_samples = 200
        #     debug_indices = np.random.choice(X_train.shape[0], num_samples, replace=False)
        #     debug_indices2 = np.random.choice(X_test.shape[0], num_samples, replace=False)
        #     X_train = X_train[debug_indices]
        #     X_test = X_test[debug_indices2]
        #     y_train = y_train[debug_indices]
        #     y_test = y_test[debug_indices2]

        #改變數
        yhat_train, x_train_transform_label = classifier.fit(
            X_train, y_train,
            predict_on_train=False  # 可改True 或 False
        )
        np.save(x_transform_path + 'x_transform_train.npy', x_train_transform_label.astype(np.float32))
        #x_test_transform = classifier.transform(X_test) #不用
        #y_test_2D = y_test.reshape(-1, 1)  # (-1表示自動計算該維度(row)大小, 指定該維度(column)數量1)
        #x_test_transform_label = np.hstack((y_test_2D, x_test_transform))  # 將x_train_transform與label拼接
        #np.save(x_transform_path + 'x_transform_test.npy', x_test_transform_label.astype(np.float32))
        # 儲存訓練好的模型
        model_path = os.path.join(output_dir, 'multirocket_model.joblib')
        dump(classifier, model_path)
        print(f"模型已儲存至 {model_path}")

        """
        若要在未來的任務中使用訓練好的模型，可以使用以下代碼來加載模型：

        from joblib import load

        model_path = 'path_to_your_saved_model/multirocket_model.joblib'  # 請確保這裡的路徑是正確的
        classifier = load(model_path)

        # 使用加載的模型進行預測
        # yhat_test = classifier.predict(X_test)
        """



"""
                                                     _ooOoo
                                                    o8888888o
                                                    88" . "88 
                                                    (| -_- |)
                                                    O\  =  /O
                                                  ___/`---'\____
                                               .'  \\|     |//  `.
                                              /  \\|||  :  |||//  \
                                             /  _||||| -:- |||||_  \
                                             |   | \\\  -  /// |   |
                                             | \_|  ''\---/''  |   |
                                             \  .-\__       __/-.  /
                                           ___`. .'  /--.--\ `. . __
                                        ."" '<  `.___\_<|>_/__.'  >'"".
                                       | | :  `- \`.;`\ _ /`;.`/ - ` : | |
                                       \  \ `-.   \_ __\ /__ _/   .-` /  /
                                  ======`-.____`-.___\_____/___.-`____.-'======
                                                     `=---='
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              佛祖保佑       永無BUG
                                        ʅ（´◔౪◔）ʃ 宥辰到此一遊 ʅ（´◔౪◔）ʃ
"""