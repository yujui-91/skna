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
from multirocket.multirocket_non_transform import MultiRocket
from utils.data_loader import read_univariate_ucr, non_109_datasets, read_univariate_ucr_skna, read_univariate_ucr_bladder, load_train_test_folder
from utils.tools import create_directory
from multirocket.logistic_regression import LogisticRegression
from tqdm import tqdm
from utils.metric import metrics
pd.set_option('display.max_columns', 500)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--problem", type=str, required=False, default="sr2500_500_1000_logistic_merge") # output data path
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000)
parser.add_argument("-t", "--num_threads", type=int, required=False, default=8)  # -1表示使用全部線程
parser.add_argument("-s", "--save", type=bool, required=False, default=True)
parser.add_argument("-v", "--verbose", type=int, required=False, default=2)
parser.add_argument("--patience", type=int, required=False, default=20, help='總訓練損失連續10次epoch保持一致的損失值或沒有再更佳的表現(損失沒下降)')  # 沒用到
parser.add_argument("--patience_lr", type=int, required=False, default=5, help='訓練學習速率連續5次保持一致的損失值或沒有再更佳的表現(損失沒下降)')  # 沒用到

arguments = parser.parse_args()

if __name__ == '__main__':
    problem = arguments.problem
    num_features = arguments.num_features
    num_threads = arguments.num_threads
    itr = arguments.iter
    save = arguments.save
    verbose = arguments.verbose

    output_path = r"V:\dunwei\MACE\dataset\multirocket_result"
    classifier_name = "MultiRocket_{}".format(num_features)
    hp_path = r"V:\dunwei\MACE\dataset\SKNA_signal\ch1\sr2500_500_1000/"
    data_folder = hp_path + r"x_transform/"

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
        print("Output Dir: {}".format(output_dir))
        print("Iteration: {}".format(itr))
        print("Problem: {}".format(problem))
        print("Number of Features: {}".format(num_features))

        train_h = pd.read_csv(hp_path + "train_h_file_name.csv")
        train_p = pd.read_csv(hp_path + "train_p_file_name.csv")
        test_h = pd.read_csv(hp_path + "test_h_file_name.csv")
        test_p = pd.read_csv(hp_path + "test_p_file_name.csv")
        h_file = pd.concat([train_h, test_h], ignore_index=True)
        p_file = pd.concat([train_p, test_p], ignore_index=True)
        h_file_list = h_file['file_name'].tolist()
        p_file_list = p_file['file_name'].tolist()
        h = np.load(data_folder + "healthy.npy")
        p = np.load(data_folder + "patient.npy")
        
        train_cm = []
        test_cm = []
        for i in tqdm(range((int((p.shape[0]/141)/1))), desc="iteration"):
            st_h = 4 * i * 141
            ed_h = st_h + (4 * 141)
            st_p = 1 * i * 141
            ed_p = st_p + (1 * 141)
        
            st_h_f = 4 * i
            ed_h_f = st_h_f + 4
            st_p_f = 1 * i
            ed_p_f = st_p_f + 1

            test = np.vstack([h[st_h:ed_h, :], p[st_p:ed_p, :]]) if ed_p <= p.shape[0] else np.vstack([h[st_h:ed_h, :], p[st_p:, :]])
            
            # 訓練資料：動態排除 test 區間
            train_h = np.vstack([h[:st_h, :], h[ed_h:, :]])
            train_p = np.vstack([p[:st_p, :], p[ed_p:, :]]) if ed_p <= p.shape[0] else p[:st_p, :]
            train = np.vstack([train_h, train_p])
            print(f"train shape: {train.shape}, test shape: {test.shape}")

            test_file_name = h_file_list[st_h_f:ed_h_f] + p_file_list[st_p_f:ed_p_f] if ed_p_f <= len(p_file_list) else h_file_list[st_h_f:ed_h_f] + p_file_list[st_p_f:]
            train_file_name = h_file_list[:st_h_f] + h_file_list[ed_h_f:] + p_file_list[:st_p_f] + p_file_list[ed_p_f:] if ed_p_f <= len(p_file_list) else h_file_list[:st_h_f] + h_file_list[ed_h_f:] + p_file_list[:st_p_f]

            test_file_name_df = pd.DataFrame(test_file_name, columns=['file_name'])
            train_file_name_df = pd.DataFrame(train_file_name, columns=['file_name'])
            test_file_name_df.to_csv(output_dir + f'test_file_name_{i + 1}.csv', index=False)
            train_file_name_df.to_csv(output_dir + f'train_file_name_{i + 1}.csv', index=False)

            X_train = train[:, 1:]
            y_train = train[:, 0]
            X_test = test[:, 1:]
            y_test = test[:, 0]

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)

            # returns ntc format, remove the last dimension
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

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

            classifier = MultiRocket(
                num_features=num_features,
                classifier="logistic",
                verbose=verbose
            )

            yhat_train = classifier.fit(
                X_train, y_train,
                predict_on_train=True  # 可改True 或 False
            )


            """
            若要在未來的任務中使用訓練好的模型，可以使用以下代碼來加載模型：

            from joblib import load

            model_path = 'path_to_your_saved_model/multirocket_model.joblib'  # 請確保這裡的路徑是正確的
            classifier = load(model_path)

            # 使用加載的模型進行預測
            # yhat_test = classifier.predict(X_test)
            """

    #### 為了顯示原本的預測值做的修改
            if yhat_train is not None:
                data_yhat_train_original = {
                    "yhat_train_original": yhat_train.ravel()
                }

                df_yhat_train_original = pd.DataFrame(data_yhat_train_original)


                if save:
                    df_yhat_train_original.to_csv(output_dir + f'yhat_train_prob_{i + 1}.csv', index=False)

                MCC_ytrain_best = 0
                thresholdi_TrainMCCbest = 0
                tn_best = 0
                fp_best = 0
                fn_best = 0
                tp_best = 0
                yhat_train_MCCbest = []
                MCC_ytrain_best = 0
                yhat_train_for_thresholdi = []




                for ii in range(1,1000):  # 為甚麼這1-1000
                    yhat_train_for_thresholdi = yhat_train.copy()
                    thresholdi = ii*0.001
                    for I in range(len(yhat_train)):

                        if yhat_train[I] >= thresholdi:
                            yhat_train_for_thresholdi[I] = 1
                        else:
                            yhat_train_for_thresholdi[I] = 0

                    tn, fp, fn, tp = confusion_matrix(y_train, yhat_train_for_thresholdi).ravel()
                    # 避免溢位 所以將數值轉換成float32
                    tn = tn.astype(np.float32)
                    tp = tp.astype(np.float32)
                    fn = fn.astype(np.float32)
                    fp = fp.astype(np.float32)
                    MCC_ytrain = round((tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5), 4)


                    if MCC_ytrain >= MCC_ytrain_best:

                        MCC_ytrain_best = MCC_ytrain.copy()
                        thresholdi_TrainMCCbest = thresholdi
                        tn_best = tn.copy()
                        fp_best = fp.copy()
                        fn_best = fn.copy()
                        tp_best = tp.copy()
                        yhat_train_MCCbest = yhat_train_for_thresholdi.copy()



    ##原本不調yhat閥值的作法        yhat_train = np.round(yhat_train)\
            yhat_train = yhat_train_MCCbest
    ####
            if yhat_train is not None:
                train_acc = round(accuracy_score(y_train, yhat_train), 4)

                # data_y_train_and_yhat_train = {
                #     "y_train": y_train.ravel(),
                #     "yhat_train": yhat_train.ravel()
                # }
                #
                # df_y_train_and_yhat_train = pd.DataFrame(data_y_train_and_yhat_train)
                #
                #
                # if save:
                #     df_y_train_and_yhat_train.to_csv(output_dir + 'results_y_train_and_yhat_train.csv', index=False)



                data_y_train_and_yhat_train = {
                    "y_train": y_train.ravel(),
                    "yhat_train_MCCbest": yhat_train.ravel()
                }

                data_thresholdi_TrainMCCbest_and_best_MCC = {
                    "thresholdi_TrainMCCbest": thresholdi_TrainMCCbest,
                    "y_train_best_MCC": MCC_ytrain_best
                }

                df_y_train_and_yhat_train_best_MCC = pd.DataFrame(data_y_train_and_yhat_train)
                df_thresholdi_TrainMCCbest = pd.DataFrame(data_thresholdi_TrainMCCbest_and_best_MCC, index=[0])

                if save:
                    df_y_train_and_yhat_train_best_MCC.to_csv(output_dir + f'y_train_label_{i + 1}.csv', index=False)
                    df_thresholdi_TrainMCCbest.to_csv(output_dir + f'yhat_thresholdi_TrainMCCbest_{i + 1}.csv', index=False)

            else:
                train_acc = -1

            # 一次載入全部測試
            # yhat_test = classifier.predict(X_test)

            # 將測試分批載入
            test_batch = 10000
            yhat_test = []
            # 遍歷所有數據，每次處理 batch_size 筆資料
            for start_idx in range(0, len(X_test), test_batch):
                end_idx = start_idx + test_batch
                X_test_batch = X_test[start_idx:end_idx]

                print(f"第{start_idx}筆-第{end_idx}筆")

                # 預測當前批次
                yhat_test_batch = classifier.predict(X_test_batch)

                # 將當前批次的預測結果添加到總的預測列表中
                yhat_test.append(yhat_test_batch)

            # 將所有批次的預測結果合併成一個列表或數組
            yhat_test = np.concatenate(yhat_test, axis=0)

            ####為了顯示原本的預測值做的修改
            if yhat_test is not None:
                data_yhat_test_original = {
                    "yhat_test_original": yhat_test.ravel()
                }

                df_yhat_test_original = pd.DataFrame(data_yhat_test_original)

                if save:
                    df_yhat_test_original.to_csv(output_dir + f'yhat_test_prob_{i + 1}.csv', index=False)


            yhat_test_for_thresholdi = []
            yhat_test_for_thresholdi = yhat_test.copy()
            for I in range(len(yhat_test)):


                if yhat_test[I] >= thresholdi_TrainMCCbest:
                    yhat_test_for_thresholdi[I] = 1
                else:
                    yhat_test_for_thresholdi[I] = 0



            yhat_test = yhat_test_for_thresholdi.copy()

            test_acc = round(accuracy_score(y_test, yhat_test), 4)

            data_y_test_and_yhat_test = {
                "y_test": y_test.ravel(),
                "yhat_test": yhat_test.ravel()
            }

            df_y_test_and_yhat_test = pd.DataFrame(data_y_test_and_yhat_test)

            if save:
                df_y_test_and_yhat_test.to_csv(output_dir + f'y_test_label_{i + 1}.csv', index=False)

            gg_test = confusion_matrix(y_test, yhat_test)
            gg_train = confusion_matrix(y_train, yhat_train)

            train_cm.append(gg_train)
            test_cm.append(gg_test)

        total_train_cm = np.sum(train_cm, axis=0)
        total_test_cm = np.sum(test_cm, axis=0)

        train_cm_df, train_metric_df = metrics(total_train_cm)
        test_cm_df, test_metric_df = metrics(total_test_cm)

        print('===========   TRAIN CONFUSION MATRIX   ===========')
        train_cm_df.to_csv(output_dir + "train_cm.csv", index=False)
        print(f"confusion matrix of train :\n{train_cm_df.to_string(index=False)}\n")

        print('===========   TEST CONFUSION MATRIX   ===========')
        test_cm_df.to_csv(output_dir + "test_cm.csv", index=False)
        print(f"confusion matrix of test :\n{test_cm_df.to_string(index=False)}\n")

        train_metric_df.to_csv(output_dir + "train_metric.csv", index=False)
        test_metric_df.to_csv(output_dir + "test_metric.csv", index=False)


    else:
        print(f'指定載入tsv路徑 {data_folder} 不存在')



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