import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov
from data_splitter import data_splitter
import pickle
import datetime
from movie_analyzer import MovieAnalyzer
from cppmodule.esn import ESN as ESNCpp
import time
import enum
from scipy.signal import find_peaks
import logging
from logging import FileHandler
from narma import NARMA
from sinsaw import SINSAW
#from sinsaw import ScalingShift
from read_speech_data import read_speech_data
from movie import Movie
import frame_diff
import argparse
import hog

np.random.seed(seed=0)

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="[%X]",
    handlers=[FileHandler(filename="log.txt")]
)
logger = logging.getLogger(__name__)

class ModuleType(enum.IntEnum):
    python = 1
    cpp = 2

class HyperParams:
    N_x = 500
    density = 0.1
    input_scale = 100.0
    rho = 0.9
    leaking_rate = 1.0

def Train(train, train_labels, train_labels_id, test, test_labels, test_labels_id, show = False, moduleType = ModuleType.cpp):
    # ESNモデル
    n_step = train.shape[1] if train.ndim == 2 else train.shape[2]
    N_x = HyperParams.N_x
    density = HyperParams.density
    input_scale = HyperParams.input_scale
    rho = HyperParams.rho
    leaking_rate = HyperParams.leaking_rate

    if moduleType == ModuleType.cpp:
        # 学習（線形回帰）
        logger.info("[Start] esn_cpp.Train")
        esn_cpp = ESNCpp(n_step, 1, N_x, density, input_scale, rho, leaking_rate)
        Y_learning = esn_cpp.Train(train, train_labels)
        logger.info("[Finish] esn_cpp.Train")

        # 学習済みモデルをファイルに保存する
        logger.info("[Start] save pickle file")
        now = datetime.datetime.now()
        model_file = "../model/"+ now.strftime('%Y%m%d_%H%M%S') + '.pickle'
        with open(model_file, mode='wb') as fo:
            pcl_w_out = esn_cpp.GetWout()

            print("python Wout")
            print(pcl_w_out.shape)
            print(pcl_w_out)
            pickle.dump(pcl_w_out, fo)
        logger.info("[Finish] save pickle file")
    
    elif moduleType == ModuleType.python:
        # 学習（線形回帰）
        logger.info("[Start] model.train")
        model = ESN(n_step, 1, N_x, density=density,
                input_scale=input_scale, rho=rho, leaking_rate=leaking_rate)
        Y_learning = model.train(train, train_labels, Tikhonov(N_x, 1, 0.0))
        logger.info("[Finish] model.train")

        # 学習済みモデルをファイルに保存する
        logger.info("[Start] save pickle file")
        now = datetime.datetime.now()
        model_file = "../model/"+ now.strftime('%Y%m%d_%H%M%S') + '.pickle'
        with open(model_file, mode='wb') as fo:
            pcl_w_out = model.get_Wout() # python Ver

            print("python Wout")
            print(pcl_w_out.shape)
            print(pcl_w_out)
            pickle.dump(pcl_w_out, fo)
        logger.info("[Finish] save pickle file")
    
    else:
        None

    return model_file

def Predict_test(train, train_labels, train_labels_id, test, test_labels, test_labels_id, model_file, moduleType = ModuleType.cpp):
    ### pickleで保存したファイルを読み込み
    with open(model_file, mode='br') as fi:
        Wout = pickle.load(fi)

    # ENSモデル
    n_step = train.shape[1] if train.ndim == 2 else train.shape[2]
    N_x = Wout.shape[1]
    density = HyperParams.density
    input_scale = HyperParams.input_scale
    rho = HyperParams.rho
    leaking_rate = HyperParams.leaking_rate

    feature = 1
    if moduleType == ModuleType.cpp:
        esn_cpp = ESNCpp(n_step, 1, N_x, density, input_scale, rho, leaking_rate)
        esn_cpp.SetWout(Wout)

        # 訓練データを使った推論
        train_Y = esn_cpp.Predict(train)
        # テストデータを使った推論
        test_Y = esn_cpp.Predict(test)

        # 結果の可視化
        plt.plot(train_labels_id, train_labels, label='original_train')
        plt.plot(test_labels_id, test_labels, label='original_test')
        # 訓練データの推論結果を可視化
        train_feature_time_id = np.arange(feature * len(train_labels_id) - len(train_labels_id)) + train_labels_id[-1] + 1
        train_labels_id = np.concatenate([train_labels_id, train_feature_time_id])
        plt.plot(train_labels_id, train_Y, label='predict_train')
        # テストデータの推論結果を可視化
        test_feature_time_id = np.arange(feature * len(test_labels_id) - len(test_labels_id)) + test_labels_id[-1] + 1
        test_labels_id = np.concatenate([test_labels_id, test_feature_time_id])
        plt.plot(test_labels_id, test_Y, label='predict_test')

        plt.legend()
        plt.xlabel("time step")
        plt.ylabel("breathing wave")
        plt.show()

    elif moduleType == ModuleType.python:
        model = ESN(n_step, 1, N_x, density=density,
                    input_scale=input_scale, rho=rho, leaking_rate= leaking_rate)
        model.set_Wout(Wout)

        # 訓練データを使った推論
        train_Y = model.predict(train, feature)
        # テストデータを使った推論
        test_Y = model.predict(test, feature)

        # 結果の可視化
        plt.plot(train_labels_id, train_labels, label='original_train')
        plt.plot(test_labels_id, test_labels, label='original_test')
        # 訓練データの推論結果を可視化
        train_feature_time_id = np.arange(feature * len(train_labels_id) - len(train_labels_id)) + train_labels_id[-1] + 1
        train_labels_id = np.concatenate([train_labels_id, train_feature_time_id])
        plt.plot(train_labels_id, train_Y, label='predict_train')
        # テストデータの推論結果を可視化
        test_feature_time_id = np.arange(feature * len(test_labels_id) - len(test_labels_id)) + test_labels_id[-1] + 1
        test_labels_id = np.concatenate([test_labels_id, test_feature_time_id])
        plt.plot(test_labels_id, test_Y, label='predict_test')

        plt.legend()
        plt.xlabel("time step")
        plt.ylabel("breathing wave")
        plt.show()

    else:
        None

def create_model(input_data, show, moduleType = ModuleType.cpp):
    logger.info("[Start] Split into training and test data")
    splitter =  data_splitter(input_data, test_size=0.3, isTrain=True)
    train, train_labels, train_labels_id, test, test_labels, test_labels_id = splitter.create_batch2(show=False, isTrain=True)
    logger.info("[Finish] Split into training and test data")

    model_file = Train(train, train_labels, train_labels_id, test, test_labels, test_labels_id, show, moduleType=moduleType)
    if show == True:
        Predict_test(train, train_labels, train_labels_id, test, test_labels, test_labels_id, model_file, moduleType=moduleType)

    print(f"created model file: {model_file}")
    return model_file

def Predict(input_data, model_file, show, moduleType = ModuleType.cpp):
    splitter =  data_splitter(input_data, test_size=0, isTrain=False)
    input = splitter.create_batch2(show=False, isTrain=False)
    if input_data.ndim == 3:
        print("python N: " + str(input.shape[0]))
        print("python N_window: " + str(input.shape[1]))
        print("python N_u: " + str(input.shape[2]))
    else:
        print("python N: " + str(input.shape[0]))
        print("python N_u: " + str(input.shape[1]))
    feature = 1# 入力データの時間幅の何倍の時間幅を予測するか

    ### pickleで保存したファイルを読み込み
    with open(model_file, mode='br') as fi:
        Wout = pickle.load(fi)

    n_step = input.shape[1] if input.ndim == 2 else input.shape[2]
    N_x = Wout.shape[1]
    density = HyperParams.density
    input_scale = HyperParams.input_scale
    rho = HyperParams.rho
    leaking_rate = HyperParams.leaking_rate

    if moduleType == ModuleType.cpp:
        # 推論
        esn_cpp = ESNCpp(n_step, Wout.shape[0], Wout.shape[1], density, input_scale, rho, leaking_rate)
        logger.info("[Start] esn_cpp.SetWout")
        esn_cpp.SetWout(Wout)
        logger.info("[Finish] esn_cpp.SetWout")
        logger.info("[Start] esn_cpp.Predict")
        Y = esn_cpp.Predict(input)
        logger.info("[Finish] esn_cpp.Predict")
        Y = Y.flatten()

        # 疑似呼吸数の計算
        peaks, _ = find_peaks(Y, distance=30)
        print(f"peakNum: {len(peaks)}")

        if show == True:
            # 結果の可視化
            plt.plot(Y, label='pridict')
            plt.xlabel("time step")
            plt.ylabel("breathing wave")
            plt.show()

    elif moduleType == ModuleType.python:
        # 推論
        model = ESN(n_step, 1, N_x, density=0.1,
                    input_scale=1.0, rho=0.9, leaking_rate= leaking_rate)
        logger.info("[Start] model.set_Wout")
        model.set_Wout(Wout)
        logger.info("[Finish] model.set_Wout")
        logger.info("[Start] model.predict")
        Y = model.predict(input, feature)
        logger.info("[Finish] model.predict")
        Y = Y.flatten()

        # 疑似呼吸数の計算
        peaks, _ = find_peaks(Y, distance=30)
        print(f"peakNum: {len(peaks)}")

        if show == True:
            # 結果の可視化
            plt.plot(Y, label='pridict')
            plt.xlabel("time step")
            plt.ylabel("breathing wave")
            plt.show()

    else:
        None

def NARMA_TEST():
    # データ長
    T = 900  # 訓練用
    T_test = 100  # 検証用

    order = 10  # NARMAモデルの次数
    dynamics = NARMA(order, a1=0.3, a2=0.05, a3=1.5, a4=0.1)
    y_init = [0] * order
    u, d = dynamics.generate_data(T + T_test, y_init)

    # 学習・テスト用情報
    train_U = u[:T].reshape(-1, 1)
    train_D = d[:T].reshape(-1, 1)
    test_U = u[T:].reshape(-1, 1)
    test_D = d[T:].reshape(-1, 1)

    # ESNモデル
    N_x = 50
    n_step = train_U.shape[1]
    n_y = train_D.shape[1]
    density = 0.15
    input_scale = 0.1
    rho = 0.9
    leaking_rate = 1.0

    #model = ESN(n_step, n_y, N_x,
    #            density=density, input_scale=input_scale, rho=rho,
    #            fb_scale=0.1, fb_seed=0)
    #Win, X, W, Wout, Wfb = model.Get()
    #Win, X, W, Wout = model.Get()

    esn_cpp = ESNCpp(1, 1, N_x, density, input_scale, rho, leaking_rate, fb_scale=0.1)
    #esn_cpp.SetW(W)
    #esn_cpp.SetWin(Win)
    #esn_cpp.SetWfb(Wfb)
    #esn_cpp.SetWout(Wout)

    # 学習（リッジ回帰）
    train_Y = esn_cpp.Train(train_U, train_D, 1e-4)
    #train_Y = model.train(train_U, train_D,
    #                      Tikhonov(N_x, train_D.shape[1], 1e-4))

    # モデル出力
    test_Y = esn_cpp.Predict(test_U)
    #test_Y = model.predict(test_U)

    # 評価（テスト誤差RMSE, NRMSE）
    RMSE = np.sqrt(((test_D - test_Y) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(test_D))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # グラフ表示用データ
    T_disp = (-100, 100)
    t_axis = np.arange(T_disp[0], T_disp[1])
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]]))
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()

def WAVE_CLASSIFICATION_TEST():
    # 訓練データ，検証データの数
    n_wave_train = 60
    n_wave_test = 40

    # 時系列入力データ生成
    period = 50
    dynamics = SINSAW(period)
    label = np.random.choice(2, n_wave_train+n_wave_test)
    u, d = dynamics.generate_data(label)
    T = period*n_wave_train

    # 訓練・検証用情報
    train_U = u[:T].reshape(-1, 1)
    train_D = d[:T]

    test_U = u[T:].reshape(-1, 1)
    test_D = d[T:]

    #ESNモデル
    N_x = 50  # リザバーのノード数

    # 出力のスケーリング関数
    #output_func = ScalingShift([0.5, 0.5], [0.5, 0.5])
    #model_python = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1,
    #        input_scale=0.2, rho=0.9, fb_scale=0.05,
    #        output_func=output_func, inv_output_func=output_func.inverse,
    #        classification = True, average_window=period)
    #Win, X, W, Wout, Wfb = model_python.Get()

    model = ESNCpp(train_U.shape[1], train_D.shape[1], N_x,
                     density=0.1, input_scale=0.2, rho=0.9, fb_scale=0.05,
                     classification=True, average_window=period,
                     y_scale=0.5, y_shift=0.5)
    #model.SetW(W)
    #model.SetWin(Win)
    #model.SetWfb(Wfb)
    #model.SetWout(Wout)

    # 学習（リッジ回帰）
    train_Y = model.Train(train_U, train_D, beta=0.1)
    #train_Y = model_python.train(train_U, train_D,
    #                      Tikhonov(N_x, train_D.shape[1], 0.1))

    # 訓練データに対するモデル出力
    test_Y = model.Predict(test_U)
    #test_Y = model_python.predict(test_U)

    # 評価（正解率, accracy）
    mode = np.empty(0, np.int32)
    for i in range(n_wave_test):
        tmp = test_Y[period*i:period*(i+1), :]  # 各ブロックの出力
        max_index = np.argmax(tmp, axis=1)  # 最大値をとるインデックス
        histogram = np.bincount(max_index)  # そのインデックスのヒストグラム
        mode = np.hstack((mode, np.argmax(histogram)))  #  最頻値

    target = test_D[0:period*n_wave_test:period,1]
    accuracy = 1-np.linalg.norm(mode.astype(np.float32)-target, 1)/n_wave_test
    print('accuracy =', accuracy)

    # グラフ表示用データ
    T_disp = (-500, 500)
    t_axis = np.arange(T_disp[0], T_disp[1])  # 時間軸
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]]))
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.1, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
    plt.ylim([-0.3, 1.3])
    plt.ylabel('Output 1')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax3 = fig.add_subplot(3, 1, 3)
    plt.plot(t_axis, disp_D[:, 1], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:, 1], color='gray', linestyle='--', label='Model')
    plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
    plt.ylim([-0.3, 1.3])
    plt.xlabel('n')
    plt.ylabel('Output 2')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()

def SPOKENDIGIT_RECOGNITION_TEST():
    # 訓練データ，検証データの取得
    train_list = [1, 2, 3, 4, 5]  # u1-u5が訓練用，残りが検証用
    train_input, train_output, train_length, train_label, \
    test_input, test_output, test_length, test_label = \
    read_speech_data(dir_name='./Lyon_decimation_128', utterance_train_list=train_list)
    print("データ読み込み完了．訓練と検証を行っています...")

    N_x_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    train_WER = np.empty(0)
    test_WER = np.empty(0)
    #bar = tqdm(total = np.sum(N_x_list))
    for N_x in N_x_list:
        print("リザバーの大きさ: %d" % N_x)

        # ESNモデル
        model = ESN(train_input.shape[1], train_output.shape[1], N_x,
                    density=0.05, input_scale=1.0e+4, rho=0.9, fb_scale=0.0)
        Win, X, W, Wout, Wfb = model.Get()

        esn_cpp = ESNCpp(train_input.shape[1], train_output.shape[1], N_x,
                         density=0.05, input_scale=1.0e+4, rho=0.9, fb_scale=0.0)

        #esn_cpp.SetW(W)
        #esn_cpp.SetWin(Win)
        #esn_cpp.SetWfb(Wfb)
        #esn_cpp.SetWout(Wout)

        ########## 訓練データに対して
        """
        # リザバー状態行列
        stateCollectMat = np.empty((0, N_x))
        for i in range(len(train_input)):
            u_in = model.Input(train_input[i])
            r_out = model.Reservoir(u_in)
            stateCollectMat = np.vstack((stateCollectMat, r_out))

        # 教師出力データ行列
        teachCollectMat = train_output

        # 学習（疑似逆行列）
        Wout = np.dot(teachCollectMat.T, np.linalg.pinv(stateCollectMat.T))
        """
        #model.train(train_input, train_output, Tikhonov(N_x, train_output.shape[1], 0.0))
        esn_cpp.Train(train_input, train_output, beta=0.0)

        # ラベル出力
        #Y_pred = np.dot(Wout, stateCollectMat.T)
        #Y_pred = model.predict(train_input)
        Y_pred = esn_cpp.Predict(train_input)
        Y_pred = Y_pred.T # 転置して、N_y * tauの行列にする
        pred_train = np.empty(0, np.int32)
        start = 0
        for i in range(len(train_length)):
            tmp = Y_pred[:,start:start+train_length[i]]  # 1つのデータに対する出力
            max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号
            histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
            pred_train = np.hstack((pred_train, np.argmax(histogram)))  # 最頻値
            start = start + train_length[i]

        # 訓練誤差(Word Error Rate, WER)
        count = 0
        for i in range(len(train_length)):
            if pred_train[i] != train_label[i]:
                count = count + 1
        print("訓練誤差： WER = %5.4lf" % (count/len(train_length)))
        train_WER = np.hstack((train_WER, count/len(train_length)))

        ########## 検証データに対して
        """
        # リザバー状態行列
        stateCollectMat = np.empty((0, N_x))
        for i in range(len(test_input)):
            u_in = model.Input(test_input[i])
            r_out = model.Reservoir(u_in)
            stateCollectMat = np.vstack((stateCollectMat, r_out))
        """

        # ラベル出力
        #Y_pred = np.dot(Wout, stateCollectMat.T)
        #Y_pred = model.predict(test_input)
        Y_pred = esn_cpp.Predict(test_input)
        Y_pred = Y_pred.T # 転置して、N_y * tauの行列にする
        pred_test = np.empty(0, dtype=np.int32)
        start = 0
        for i in range(len(test_length)):
            tmp = Y_pred[:,start:start+test_length[i]]  # 1つのデータに対する出力
            max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号
            histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
            pred_test = np.hstack((pred_test, np.argmax(histogram)))  # 最頻値
            start = start + test_length[i]

        # 検証誤差(WER)
        count = 0
        for i in range(len(test_length)):
            if pred_test[i] != test_label[i]:
                count = count + 1
        print("検証誤差： WER = %5.4lf" % (count/len(test_length)))
        test_WER = np.hstack((test_WER, count/len(test_length)))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))

    plt.plot(N_x_list, train_WER, marker='o', fillstyle='none',
             markersize=8, color='k', label='Training')
    plt.plot(N_x_list, test_WER, marker='s',
             markersize=8, color='k', label='Testing')
    plt.xticks(N_x_list)
    plt.xlabel("Size of reservoir")
    plt.ylabel("WER")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    plt.show()

def main():
    #NARMA_TEST()
    #WAVE_CLASSIFICATION_TEST()
    #SPOKENDIGIT_RECOGNITION_TEST()

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--getimages', action='store_true')
    parser.add_argument('-d', '--diffframe', action='store_true')
    parser.add_argument('-o', '--hog', action='store_true')
    args = parser.parse_args()

    if args.getimages:
        movie_file = "/root/app/data/input_video.mp4"
        movie = Movie()
        movie.Read(movie_file)
        movie.CreateFrames(num_cut=30, save=True, binarize=True)

        return
    elif args.diffframe:
        frame_diff.get_diff()
        return
    elif args.hog:
        hog.get_hog()
        return

    start = time.perf_counter() #計測開始
    isTrain = True
    moduleType = ModuleType.cpp
    show = True

    if moduleType == ModuleType.cpp:
        logger.info("Use cpp module")
    else:
        logger.info("Use python module")

    if isTrain == True:
        # train
        logger.info("[Start] Train")
        logger.info("[Start] Read data and create frame images")
        movie_file = "/root/app/data/input_video.mp4"
        analyzer = MovieAnalyzer(movie_file)
        input_data = analyzer.GetColor(show=show, save=False, isTrain=isTrain)
        #input_data = analyzer.GetGray(show=show, save=False, isTrain=isTrain)
        logger.info("[Finish] Read data and create frame images")

        # モデル評価用の推論処理を呼び出す
        model_file = create_model(input_data, show=show, moduleType=moduleType)
        logger.info("[Finish] Train")
    else:
        # predict
        logger.info("[Start] Predict")
        logger.info("[Start] Read data and create frame images")
        movie_file = "/root/app/data/input_video.mp4"
        analyzer = MovieAnalyzer(movie_file)
        input_data = analyzer.GetColor(show=show, save=False, isTrain=isTrain)
        #input_data = analyzer.GetGray(show=show, save=False, isTrain=isTrain)
        logger.info("[Finish] Read data and create frame images")

        model_file = "/root/app/model/20250503_084922.pickle"
        Predict(input_data, model_file, show=show, moduleType=moduleType)
        logger.info("[Finish] Predict")

    end = time.perf_counter() #計測終了
    print('{:.2f}'.format((end-start)/60))

if __name__ == '__main__':
    main()