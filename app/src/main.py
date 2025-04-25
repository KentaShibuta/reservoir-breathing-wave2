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

def Train(train, train_labels, train_labels_id, test, test_labels, test_labels_id, show = False, moduleType = ModuleType.cpp):
    # ESNモデル
    N_x = 500
    n_step = train.shape[1] if train.ndim == 2 else train.shape[2]
    density = 0.1
    input_scale = 1.0
    rho = 0.9
    leaking_rate = 1.0

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
    N_x = Wout.shape[1]
    n_step = train.shape[1] if train.ndim == 2 else train.shape[2]
    density = 0.1
    input_scale = 1.0
    rho = 0.9
    leaking_rate = 1.0

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
    # バイナリの読み込み
    #print(f"input file: {input_file}")
    #bindata = self.read_binary(input_file)
    #input_data = np.array(bindata)

    logger.info("[Start] Split into training and test data")
    splitter =  data_splitter(input_data, test_size=0.3, isTrain=True)
    train, train_labels, train_labels_id, test, test_labels, test_labels_id = splitter.create_batch(show=False, isTrain=True)
    logger.info("[Finish] Split into training and test data")

    model_file = Train(train, train_labels, train_labels_id, test, test_labels, test_labels_id, show, moduleType=moduleType)
    if show == True:
        Predict_test(train, train_labels, train_labels_id, test, test_labels, test_labels_id, model_file, moduleType=moduleType)

    print(f"created model file: {model_file}")
    return model_file

def Predict(input_data, model_file, show, moduleType = ModuleType.cpp):
    splitter =  data_splitter(input_data, test_size=0, isTrain=False)
    input = splitter.create_batch(show=False, isTrain=False)
    print("python N: " + str(input.shape[0]))
    print("python N_window: " + str(input.shape[1]))
    print("python N_u: " + str(input.shape[2]))
    feature = 1# 入力データの時間幅の何倍の時間幅を予測するか

    ### pickleで保存したファイルを読み込み
    with open(model_file, mode='br') as fi:
        Wout = pickle.load(fi)

    N_x = Wout.shape[1]
    n_step = input.shape[1] if input.ndim == 2 else input.shape[2]
    density = 0.1
    input_scale = 1.0
    rho = 0.9
    leaking_rate = 1.0

    if moduleType == ModuleType.cpp:
        # 推論
        esn_cpp = ESNCpp(input.shape[2], Wout.shape[0], Wout.shape[1], density, input_scale, rho, leaking_rate)
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

def main():
    """
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
        movie_file = "/root/app/data/20241227_sophie_1_stabilization.mp4"
        analyzer = MovieAnalyzer(movie_file)
        input_data = analyzer.GetColor(show=show, save=False, isTrain=isTrain)
        logger.info("[Finish] Read data and create frame images")

        # モデル評価用の推論処理を呼び出す
        model_file = create_model(input_data, show=show, moduleType=moduleType)
        logger.info("[Finish] Train")
    else:
        # predict
        logger.info("[Start] Predict")
        logger.info("[Start] Read data and create frame images")
        movie_file = "/root/app/data/20241227_sophie_1_stabilization.mp4"
        analyzer = MovieAnalyzer(movie_file)
        input_data = analyzer.GetColor(show=show, save=False, isTrain=isTrain)
        logger.info("[Finish] Read data and create frame images")

        model_file = "/root/app/model/20250405_013307.pickle"
        Predict(input_data, model_file, show=show, moduleType=moduleType)
        logger.info("[Finish] Predict")

    end = time.perf_counter() #計測終了
    print('{:.2f}'.format((end-start)/60))
    """

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

    esn_cpp = ESNCpp(1, 1, N_x, density, input_scale, rho, leaking_rate)
    #model = ESN(n_step, n_y, N_x,
    #            density=density, input_scale=input_scale, rho=rho)#,
                #fb_scale=0.1, fb_seed=0)

    # 学習（リッジ回帰）

    train_DT = train_D.reshape(-1)
    train_Y = esn_cpp.Train(train_U, train_DT)
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

if __name__ == '__main__':
    main()