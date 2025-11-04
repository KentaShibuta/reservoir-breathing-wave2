import numpy as np
import matplotlib.pyplot as plt
from cppmodule.esn import ESN as ESNCpp

FS = 30
def SaveResult(inputX, inputY, predictY, timeSeriesFileName, errorFileName):
    T = len(inputX) # 時間軸を作る
    time = np.arange(T) / FS  # 秒単位

    plt.figure(figsize=(12, 4))
    plt.plot(time, inputY, color='blue', label='original')
    plt.plot(time, predictY, color='red', label='predict')
    plt.xlabel("time")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.savefig(timeSeriesFileName, dpi=300)
    plt.close()

    # 訓練誤差
    error_train = abs(inputY - predictY.reshape(-1))
    plt.figure(figsize=(12, 4))
    plt.plot(time, error_train, color='blue')
    plt.xlabel("time")
    plt.ylabel("y")
    plt.grid(True)

    plt.savefig(errorFileName, dpi=300)
    plt.close()

def RunTimeSeriesPrediction():
    trainData = np.loadtxt('/root/app/data/LINE_ALBUM_今日のソフィ☀️🍉ver.7_250922_1/ROI_series_filtered.csv', delimiter=',', dtype='float32')
    trainX = trainData[:-1]
    trainY = trainData[1:]

    #testData = np.loadtxt('/root/app/data/IMG_1693/ROI_series_filtered.csv', delimiter=',', dtype='float32')
    #testData = np.loadtxt('/root/app/data/2990AF23-A92C-4325-BA7C-92241E243CD9_stabilization/ROI_series_filtered.csv', delimiter=',', dtype='float32')
    #testData = np.loadtxt('/root/app/data/IMG_1873/ROI_series_filtered.csv', delimiter=',', dtype='float32')
    testData = np.loadtxt('/root/app/data/IMG_2047/ROI_series_filtered.csv', delimiter=',', dtype='float32')
    
    testX = testData[:-1]
    testY = testData[1:]

    print(f"trainX size: {trainX.shape}")
    print(f"trainY size: {trainY.shape}")
    print(f"testX size: {testX.shape}")
    print(f"testY size: {testY.shape}")

    np.savetxt('/root/app/data/timeseries/trainX.csv', trainX, delimiter=',')
    np.savetxt('/root/app/data/timeseries/trainY.csv', trainY, delimiter=',')
    np.savetxt('/root/app/data/timeseries/testX.csv', testX, delimiter=',')
    np.savetxt('/root/app/data/timeseries/testY.csv', testY, delimiter=',')

    SaveResult(trainX, trainY, testY, "/root/app/data/timeseries/trainData_predict.png", "/root/app/data/timeseries/trainData_error.png")

    # ESNモデル
    N_x = 300
    density = 0.15
    input_scale = 0.5
    rho = 0.9
    leaking_rate = 0.1

    esn_cpp = ESNCpp(1, 1, N_x,
                            density=density, input_scale=input_scale, rho=rho, fb_scale=0.0, leaking_rate=leaking_rate,
                            classification=False, average_window=0,
                            y_scale=0.5, y_shift=0.5, reset_reservoir_state=False,
                            two_class_weight=False, positive_weight=1.0, negative_weight=1.0, plot_x=False, plot_n_max=0,
                            write_log=False)

    # 学習（リッジ回帰）
    train_Y = esn_cpp.Train(trainX.reshape(-1, 1), trainY.reshape(-1, 1), 1e-1)

    # 訓練データを使った推論
    predict_train_Y = esn_cpp.Predict(trainX.reshape(-1, 1))
    SaveResult(trainX, trainY, predict_train_Y, "/root/app/data/timeseries/trainData_predict.png", "/root/app/data/timeseries/trainData_error.png")

    ### テストデータを使った推論
    predict_test_Y = esn_cpp.Predict(testX.reshape(-1, 1))
    SaveResult(testX, testY, predict_test_Y, "/root/app/data/timeseries/testData_predict.png", "/root/app/data/timeseries/testData_error.png")


def main():
    RunTimeSeriesPrediction()

if __name__ == '__main__':
    main()