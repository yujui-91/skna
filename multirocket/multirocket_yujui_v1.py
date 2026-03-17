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
from utils.data_loader_yujui_1 import load_npy
from utils.tools import create_directory
from multirocket.logistic_regression import LogisticRegression
pd.set_option('display.max_columns', 500)


parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--datapath", type=str, required=False, default=r"V:\Oswin\MACE\dataset\SKNA_signal\window\ch1/")  # 學長檔案路徑
parser.add_argument("-d", "--datapath", type=str, required=False, default=r"D:\M143020071\raw_data_result\SKNA_signal\ch1/") #路徑
parser.add_argument("-p", "--problem", type=str, required=False, default="sr10000_1000_2000_20s_ECG_signal_rpeak_5_10min_longer_100pts_mr_20win")#路徑大部分都改這 
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000) # 不能更改, original 50000
parser.add_argument("-t", "--num_threads", type=int, required=False, default=10)  # -1表示使用全部線程
parser.add_argument("-s", "--save", type=bool, required=False, default=True)
parser.add_argument("-v", "--verbose", type=int, required=False, default=2)#0:安靜 1:重要資訊 2:全印
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

    output_path = r"D:\M143020071\multirocket_result" 
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

        if os.path.exists(data_folder + 'X_all.npy'):
            print("Loading merged data from npy...")
            X_all = np.load(data_folder + 'X_all.npy')
            y_all = np.load(data_folder + 'y_all.npy')
        else:
            print("Reading original CSVs...")
            train_h_df = pd.read_csv(data_folder + "train_h_file_name.csv").iloc[:, 0]
            train_p_df = pd.read_csv(data_folder + "train_p_file_name.csv").iloc[:, 0]
            test_h_df = pd.read_csv(data_folder + "test_h_file_name.csv").iloc[:, 0]
            test_p_df = pd.read_csv(data_folder + "test_p_file_name.csv").iloc[:, 0]

            all_h_files = pd.concat([train_h_df, test_h_df], axis=0, ignore_index=True)
            all_p_files = pd.concat([train_p_df, test_p_df], axis=0, ignore_index=True)

            print(f"Total Healthy files: {len(all_h_files)}, Total Patient files: {len(all_p_files)}")

            X_h, y_h, wpp_h, X_p, y_p, wpp_p = load_npy(
                data_folder + "non_mace_zscore/",
                data_folder + "mace_zscore/",
                all_h_files, 
                all_p_files
            )
    
            print("Merging Healthy and Patient datasets into X_all...")
            
            X_all = np.concatenate((X_h, X_p), axis=0)
            y_all = np.concatenate((y_h, y_p), axis=0)
            window_per_person_all = wpp_h + wpp_p 
            file_name_all = pd.concat([all_h_files, all_p_files], axis=0, ignore_index=True)
            
            h_wpp_df = pd.DataFrame({
                "file_name": all_h_files,
                "window_per_person": wpp_h
            })
            h_wpp_df.to_csv(data_folder + "h_window_per_person.csv", index=False)
            print(f"Healthy window info saved to {data_folder}h_window_per_person.csv")

            p_wpp_df = pd.DataFrame({
                "file_name": all_p_files,    
                "window_per_person": wpp_p
            })
            p_wpp_df.to_csv(data_folder + "p_window_per_person.csv", index=False)
            print(f"patient window info saved to {data_folder}p_window_per_person.csv")         


            np.save(data_folder + 'X_all.npy', X_all)  
            np.save(data_folder + 'y_all.npy', y_all)
            print(f"Merged data saved. Shape: {X_all.shape}")
            print(f"Window info saved to {data_folder}all_window_per_person.csv")
       
        encoder = LabelEncoder()
        y_all = encoder.fit_transform(y_all) 

        # ==========================================
        # 2. 維度檢查與調整 (Reshape)
        # ==========================================
        # MultiRocket輸入格式(樣本數 N, 時間點 T) 的 2D 矩陣
        if len(X_all.shape) == 3:
            print(f"Reshaping X_all from {X_all.shape} to (N, T)...")
            X_all = X_all.reshape((X_all.shape[0], X_all.shape[1]))
        
        print(f"Final Input Shape for MultiRocket: {X_all.shape}")

        nb_classes = len(np.unique(y_all)) 
        classifier = MultiRocket(
            num_features=num_features,
            classifier="logistic",
            verbose=verbose
        )

        print("Starting MultiRocket Processing (Full Dataset)...")
        print("注意：這一步驟是為了產生特徵 (Features)，準確度數值僅供參考。")

        # x_all_transform要的 "轉換後特徵矩陣"
        yhat, x_all_transform = classifier.fit(
            X_all, 
            y_all,
            predict_on_train=False 
        )

        #  儲存結果
        transform_save_path = x_transform_path + 'x_transform_all.npy'
        np.save(transform_save_path, x_all_transform.astype(np.float32))
        print(f"特徵矩陣已儲存至: {transform_save_path}")
        
        label_save_path = x_transform_path + 'y_transform_all.npy'
        np.save(label_save_path, y_all)
        print(f"對應標籤已儲存至: {label_save_path}")

        model_path = os.path.join(output_dir, 'multirocket_model_full.joblib')
        dump(classifier, model_path)
        print(f"完整模型已儲存至 {model_path}")
        print("Processing Complete.")
        

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