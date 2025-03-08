import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov
from data_splitter import data_splitter
import pickle
import datetime
from movie_analyzer import MovieAnalyzer
from esn import ESN as ESNCpp
import time

np.random.seed(seed=0)

def Train(train, train_labels, train_labels_id, test, test_labels, test_labels_id, show = False):
    # ENSモデル
    N_x = 500
    n_step = train.shape[1] if train.ndim == 2 else train.shape[2]
    model = ESN(n_step, train_labels.shape[0], N_x, density=0.1,
                input_scale=1.0, rho=0.9, leaking_rate= 1.0)
    
    # 学習（線形回帰）
    Y_learning = model.train(train, train_labels, Tikhonov(N_x, 1, 0.0))

    # 学習済みモデルをファイルに保存する
    now = datetime.datetime.now()
    model_file = "../model/"+ now.strftime('%Y%m%d_%H%M%S') + '.pickle'
    with open(model_file, mode='wb') as fo:
        pickle.dump(model.get_Wout(), fo)
    
    """
    if show == True:
        # 結果の可視化
        # オリジナルデータを可視化
        plt.plot(train_labels_id, train_labels, label='original_train')
        plt.plot(test_labels_id, test_labels, label='original_test')
        # 訓練データの推論結果を可視化
        plt.plot(train_labels_id, Y_learning, label='learing_train')

        plt.legend()

        plt.xlabel("time step")
        plt.ylabel("Y")
        plt.show()
    """

    return model_file

def Predict_test(train, train_labels, train_labels_id, test, test_labels, test_labels_id, model_file):
    ### pickleで保存したファイルを読み込み
    with open(model_file, mode='br') as fi:
        Wout = pickle.load(fi)

    # ENSモデル
    N_x = Wout.shape[1]
    n_step = train.shape[1] if train.ndim == 2 else train.shape[2]
    model = ESN(n_step, train_labels.shape[0], N_x, density=0.1,
                input_scale=1.0, rho=0.9, leaking_rate= 1.0)
    model.set_Wout(Wout)

    # 学習済みモデルを使って予測
    # 訓練データと使った推論
    feature = 1# 入力データの時間幅の何倍の時間幅を予測するか
    train_Y = model.predict(train, feature)
    # テストデータと使った推論
    test_Y = model.predict(test, feature)

    # 結果の可視化
    # オリジナルデータを可視化
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
    plt.ylabel("Y")
    plt.show()

def create_model(input_data, show):
    # バイナリの読み込み
    #print(f"input file: {input_file}")
    #bindata = self.read_binary(input_file)
    #input_data = np.array(bindata)

    splitter =  data_splitter(input_data, test_size=0.3, isTrain=True)
    train, train_labels, train_labels_id, test, test_labels, test_labels_id = splitter.create_batch(show=False, isTrain=True)

    model_file = Train(train, train_labels, train_labels_id, test, test_labels, test_labels_id, show)
    Predict_test(train, train_labels, train_labels_id, test, test_labels, test_labels_id, model_file)

    print(f"created model file: {model_file}")
    return model_file

def Predict(input_data, model_file):
    splitter =  data_splitter(input_data, test_size=0, isTrain=False)
    input = splitter.create_batch(show=False, isTrain=False)
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
    model = ESN(n_step, input.shape[0], N_x, density=0.1,
                input_scale=1.0, rho=0.9, leaking_rate= leaking_rate)
    model.set_Wout(Wout)

    # call python module
    #Y = model.predict(input, feature)
    #np.savetxt('./predict_py.csv', Y, delimiter=',')

    # call c++ module
    #model_Win, model_x, model_W, model_Wout =  model.Get()
    #esn_cpp = ESNCpp(input, model_Win, model_W, model_Wout, model_x, leaking_rate)
    esn_cpp = ESNCpp(input.shape[2], Wout.shape[0], Wout.shape[1], density, input_scale, rho, leaking_rate)
    esn_cpp.SetInput(input, Wout)
    Y = esn_cpp.Predict()
    np.savetxt('./predict_cpp.csv', Y, delimiter=',')

    # 結果の可視化
    # オリジナルデータを可視化
    plt.plot(Y, label='pridict')

    plt.xlabel("time step")
    plt.ylabel("Y")
    plt.show()

def main():
    start = time.perf_counter() #計測開始
    isTrain = False

    if isTrain == True:
        # train
        movie_file = "/root/app/data/20241227_sophie_1_stabilization.mp4"
        analyzer = MovieAnalyzer(movie_file)
        input_data = analyzer.GetColor(show=True, save=False, isTrain=isTrain)

        # モデル評価用の推論処理を呼び出す
        model_file = create_model(input_data, False)
    else:
        # predict
        movie_file = "/root/app/data/20241227_sophie_1_stabilization.mp4"
        analyzer = MovieAnalyzer(movie_file)
        input_data = analyzer.GetColor(show=True, save=False, isTrain=isTrain)

        model_file = "/root/app/model/20250305_181243.pickle"
        Predict(input_data, model_file)

    end = time.perf_counter() #計測終了
    print('{:.2f}'.format((end-start)/60))


if __name__ == '__main__':
    main()