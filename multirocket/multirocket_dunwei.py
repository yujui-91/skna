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
parser.add_argument("-d", "--datapath", type=str, required=False, default=r"V:\dunwei\MACE\dataset\SKNA_signal\ch1/") # input data path
parser.add_argument("-p", "--problem", type=str, required=False, default="sr2500_500_1000")  # skna overactiv_bladder ori_1channel
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000) # 不能更改
parser.add_argument("-t", "--num_threads", type=int, required=False, default=8)  # -1表示使用全部線程
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
    output_path = r"V:\dunwei\MACE\dataset\multirocket_result"
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

            X_train, y_train, train_window_per_person = load_npy(data_folder + "healthy/",
                                                                 data_folder + "patient/",
                                                                 train_h_df,
                                                                 train_p_df)

            X_test, y_test, test_window_per_person = load_npy(data_folder + "healthy/",
                                                              data_folder + "patient/",
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
            np.save(data_folder + 'X_test.npy', X_test)
            np.save(data_folder + 'y_test.npy', y_test)
        
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
        # nb_classes = len(np.unique(np.concatenate(y_train, axis=0)))
        classifier = MultiRocket(
            num_features=num_features,
            classifier="logistic",
            verbose=verbose
        )

        yhat_train, x_train_transform_label = classifier.fit(
            X_train, y_train,
            predict_on_train=True  # 可改True 或 False
        )
        np.save(x_transform_path + 'x_transform_train.npy', x_train_transform_label.astype(np.float32))
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

#### 為了顯示原本的預測值做的修改
        if yhat_train is not None:
            data_yhat_train_original = {
                "yhat_train_original": yhat_train.ravel()
            }

            df_yhat_train_original = pd.DataFrame(data_yhat_train_original)


            if save:
                df_yhat_train_original.to_csv(output_dir + 'results_yhat_train_original.csv', index=False)

            MCC_ytrain_best = 0
            thresholdi_TrainMCCbest = 0
            tn_best = 0
            fp_best = 0
            fn_best = 0
            tp_best = 0
            yhat_train_MCCbest = []
            MCC_ytrain_best = 0
            yhat_train_for_thresholdi = []




            for i in range(1,1000):  # 為甚麼這1-1000
                yhat_train_for_thresholdi = yhat_train.copy()
                thresholdi = i*0.001
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
                df_y_train_and_yhat_train_best_MCC.to_csv(output_dir + 'y_train_and_yhat_train_best_MCC.csv', index=False)
                df_thresholdi_TrainMCCbest.to_csv(output_dir + 'yhat_thresholdi_TrainMCCbest.csv', index=False)

        else:
            train_acc = -1

        # 一次載入全部測試
        # yhat_test = classifier.predict(X_test)

        # 將測試分批載入
        test_batch = 10000
        yhat_test = []
        x_transform_test = []
        # 遍歷所有數據，每次處理 batch_size 筆資料
        for start_idx in range(0, len(X_test), test_batch):
            end_idx = start_idx + test_batch
            X_test_batch = X_test[start_idx:end_idx]

            print(f"第{start_idx}筆-第{end_idx}筆")

            # 預測當前批次
            yhat_test_batch, x_transform_batch = classifier.predict(X_test_batch)

            # 將當前批次的預測結果添加到總的預測列表中
            yhat_test.append(yhat_test_batch)

            x_transform_test.append(x_transform_batch)

        # 將所有批次的預測結果合併成一個列表或數組
        yhat_test = np.concatenate(yhat_test, axis=0)
        x_transform_test = np.concatenate(x_transform_test, axis=0)
        y_test_2D = y_test.reshape(-1, 1)  # (-1表示自動計算該維度(row)大小, 指定該維度(column)數量1)
        x_test_transform_label = np.hstack((y_test_2D, x_transform_test))  # 將x_train_transform與label拼接
        np.save(x_transform_path + 'x_transform_test.npy', x_test_transform_label.astype(np.float32))

        ####為了顯示原本的預測值做的修改
        if yhat_test is not None:
            data_yhat_test_original = {
                "yhat_test_original": yhat_test.ravel()
            }

            df_yhat_test_original = pd.DataFrame(data_yhat_test_original)

            if save:
                df_yhat_test_original.to_csv(output_dir + 'results_yhat_test_original.csv', index=False)



##原本不調yhat閥值的作法        yhat_test = np.round(yhat_test)

##調閥值的寫法

        yhat_test_for_thresholdi = []
        yhat_test_for_thresholdi = yhat_test.copy()
        for I in range(len(yhat_test)):


            if yhat_test[I] >= thresholdi_TrainMCCbest:
                yhat_test_for_thresholdi[I] = 1
            else:
                yhat_test_for_thresholdi[I] = 0



        yhat_test = yhat_test_for_thresholdi.copy()

##
        ####




        test_acc = round(accuracy_score(y_test, yhat_test), 4)





        # get cpu information
        # physical_cores = psutil.cpu_count(logical=False)
        # logical_cores = psutil.cpu_count(logical=True)
        # cpu_freq = psutil.cpu_freq()
        # max_freq = cpu_freq.max
        # min_freq = cpu_freq.min
        # memory = np.round(psutil.virtual_memory().total / 1e9)

        gg_test = confusion_matrix(y_test, yhat_test)
        gg_train = confusion_matrix(y_train, yhat_train)

        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, yhat_test).ravel()
        tn, fp, fn, tp = confusion_matrix(y_train, yhat_train).ravel()

        print('===========   TRAIN CONFUSION MATRIX   ===========')
        cm = {
            "predict/actual": ["Positive", "Negative"],
            "Positive": [tp, fn],
            "Negative": [fp, tn]
        }

        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(output_dir + "train_cm.csv", index=False)
        # print(gg_train)
        print(f"confusion matrix of train :\n{cm_df.to_string(index=False)}\n")
        print(f'TP_train:{tp}')
        print(f'FP_train:{fp}')
        print(f'FN_train:{fn}')
        print(f'TN_train:{tn}')

        print('===========   TEST CONFUSION MATRIX   ===========')
        cm_t = {
            "predict/actual": ["Positive", "Negative"],
            "Positive": [tp_t, fn_t],
            "Negative": [fp_t, tn_t]
        }

        cm_t_df = pd.DataFrame(cm_t)
        cm_t_df.to_csv(output_dir + "test_cm.csv", index=False)
        # print(gg_test)
        print(f"confusion matrix of test :\n{cm_t_df.to_string(index=False)}\n")
        print(f'TP_test:{tp_t}')
        print(f'FP_test:{fp_t}')
        print(f'FN_test:{fn_t}')
        print(f'TN_test:{tn_t}')

#train
        # 避免溢位 所以將數值轉換成float32
        tn = tn.astype(np.float32)
        tp = tp.astype(np.float32)
        fn = fn.astype(np.float32)
        fp = fp.astype(np.float32)

        sensitivity = round(tp/(tp+fn), 4)
        specificity = round(tn/(tn+fp), 4)
        PPV = round(tp/(tp+fp), 4)
        NPV = round(tn/(tn+fn), 4)
        KAPPA = round((2*((tp*tn)-(fp*fn)))/(((tp+fn)*(fn+tn))+((fp+tn)*(tp+fp))), 4)
        F2_score = round((2*NPV*specificity)/(NPV+specificity), 4)
        MCC = round((tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5), 4)

        metric = {
            "Accuracy": ["Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC"],
            train_acc: [sensitivity, specificity, PPV, NPV, KAPPA, MCC]
        }

        metric_df = pd.DataFrame(metric)
        metric_df.to_csv(output_dir + "train_metric.csv", index=False)

# test
        # 避免溢位 所以將數值轉換成float32
        tn_t = tn_t.astype(np.float32)
        tp_t = tp_t.astype(np.float32)
        fn_t = fn_t.astype(np.float32)
        fp_t = fp_t.astype(np.float32)

        sensitivity_t = round(tp_t / (tp_t + fn_t), 4)
        specificity_t = round(tn_t / (tn_t + fp_t), 4)
        PPV_t = round(tp_t / (tp_t + fp_t), 4)
        NPV_t = round(tn_t / (tn_t + fn_t), 4)
        KAPPA_t = round((2 * ((tp_t * tn_t) - (fp_t * fn_t))) / (((tp_t + fn_t) * (fn_t + tn_t)) + ((fp_t + tn_t) * (tp_t + fp_t))), 4)
        F2_score_t = round((2 * NPV_t * specificity_t) / (NPV_t + specificity_t), 4)
        MCC_t = round((tp_t * tn_t - fp_t * fn_t) / (((tp_t + fp_t) * (tp_t + fn_t) * (tn_t + fp_t) * (tn_t + fn_t)) ** 0.5), 4)

        metric_t = {
            "Accuracy": ["Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC"],
            test_acc: [sensitivity_t, specificity_t, PPV_t, NPV_t, KAPPA_t, MCC_t]
        }

        metric_t_df = pd.DataFrame(metric_t)
        metric_t_df.to_csv(output_dir + "test_metric.csv", index=False)

        df_metrics = pd.DataFrame(data=np.zeros((1, 44), dtype=np.float32), index=[0],
                                  columns=['timestamp', 'itr', 'classifier',
                                           'num_features','num_kernels', #'total_train_loss',#'total_val_loss',
                                           'dataset',#'learning_rate',#'epoch',
                                           'train_acc', 'train_time',
                                           'test_acc', 'test_time',
                                           'generate_kernel_time',
                                           'apply_kernel_on_train_time',
                                           'apply_kernel_on_test_time',
                                           'train_transform_time',
                                           'test_transform_time',
                                           'machine', 'processor',
                                           'physical_cores',
                                           "logical_cores",
                                           'max_freq', 'min_freq', 'memory',
                                           'train_tn','train_fp','train_fn','train_tp',
                                           'test_tn', 'test_fp', 'test_fn', 'test_tp',
                                           'train_sensitivity','train_specificity','train_PPV',
                                           'train_NPV','train_KAPPA','train_F2_score','train_MCC',
                                           'test_sensitivity','test_specificity','test_PPV',
                                           'test_NPV','test_KAPPA','test_F2_score','test_MCC'])
        df_metrics["timestamp"] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        df_metrics["itr"] = itr
        df_metrics["classifier"] = classifier_name
        df_metrics["num_features"] = num_features
        df_metrics["num_kernels"] = classifier.num_kernels
        df_metrics["dataset"] = problem
 #       df_metrics["learning_rate"] = LogisticRegression.learning_rate
 #       df_metrics["epoch"] = LogisticRegression.fit.epoch
 #       df_metrics["total_train_loss"] = LogisticRegression.fit.total_train_loss
 #       df_metrics["total_val_loss"] = LogisticRegression.fit.total_val_loss
        df_metrics["train_acc"] = train_acc
        df_metrics["train_time"] = classifier.train_duration
        df_metrics["test_acc"] = test_acc
        df_metrics["test_time"] = classifier.test_duration
        df_metrics["generate_kernel_time"] = classifier.generate_kernel_duration
        df_metrics["apply_kernel_on_train_time"] = classifier.apply_kernel_on_train_duration
        df_metrics["apply_kernel_on_test_time"] = classifier.apply_kernel_on_test_duration
        df_metrics["train_transform_time"] = classifier.train_transforms_duration
        df_metrics["test_transform_time"] = classifier.test_transforms_duration
        df_metrics["machine"] = socket.gethostname()
        df_metrics["processor"] = platform.processor()
        # df_metrics["physical_cores"] = physical_cores
        # df_metrics["logical_cores"] = logical_cores
        # df_metrics["max_freq"] = max_freq
        # df_metrics["min_freq"] = min_freq
        # df_metrics["memory"] = memory
        df_metrics["train_tn"] = tn
        df_metrics["train_fp"] = fp
        df_metrics["train_fn"] = fn
        df_metrics["train_tp"] = tp
        df_metrics["test_tn"] = tn_t
        df_metrics["test_fp"] = fp_t
        df_metrics["test_fn"] = fn_t
        df_metrics["test_tp"] = tp_t
        df_metrics["train_sensitivity"] = sensitivity
        df_metrics["train_specificity"] = specificity
        df_metrics["train_PPV"] = PPV
        df_metrics["train_NPV"] = NPV
        df_metrics["train_KAPPA"] = KAPPA
        df_metrics["train_F2_score"]= F2_score
        df_metrics["train_MCC"] = MCC
        df_metrics["test_sensitivity"] = sensitivity_t
        df_metrics["test_specificity"] = specificity_t
        df_metrics["test_PPV"] = PPV_t
        df_metrics["test_NPV"] = NPV_t
        df_metrics["test_KAPPA"] = KAPPA_t
        df_metrics["test_F2_score"] = F2_score_t
        df_metrics["test_MCC"] = MCC_t

        print('===========   RESULTS   ===========')
        # print(df_metrics)
        print(f"metric of train :\n{metric_df.to_string(index=False)}\n")
        print(f"metric of test :\n{metric_t_df.to_string(index=False)}\n")

        if save:
            df_metrics.to_csv(output_dir + 'results_train.csv', index=False)


        data_y_test_and_yhat_test = {
            "y_test": y_test.ravel(),
            "yhat_test": yhat_test.ravel()
        }

        df_y_test_and_yhat_test = pd.DataFrame(data_y_test_and_yhat_test)

        # print(df_y_test_and_yhat_test)  # 不需要

        if save:
            df_y_test_and_yhat_test.to_csv(output_dir + 'results_y_test_and_yhat_test.csv', index=False)
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