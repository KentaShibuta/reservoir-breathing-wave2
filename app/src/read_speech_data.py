import numpy as np
import glob
import os
from scipy.io import loadmat

# 音声信号を前処理したデータ(コクリアグラム)の読み込み
def read_speech_data(dir_name, utterance_train_list):
    ''' 
    :入力：データファイル(.mat)の入っているディレクトリ名(dir_name)
    :出力：入力データ(input_data)と教師データ(teacher_data)
    '''
    # .matファイルのみを取得
    data_files = glob.glob(os.path.join(dir_name, '*.mat'))

    # データに関する情報
    n_channel = 77  # チャネル数
    n_label = 10  # ラベル数(digitの数)

    # 初期化
    train_input = np.empty((0, n_channel))  # 教師入力
    train_output = np.empty((0, n_label))  # 教師出力
    train_length = np.empty(0, np.int32)  # データ長
    train_label = np.empty(0, np.int32)  # 正解ラベル
    test_input = np.empty((0, n_channel))  # 教師入力
    test_output = np.empty((0, n_label))  # 教師出力
    test_length = np.empty(0, np.int32)  # データ長
    test_label = np.empty(0, np.int32)  # 正解ラベル
    
    # データ読み込み
    if len(data_files) > 0:
        print("%d files in %s を読み込んでいます..." \
              % (len(data_files), dir_name))
        for each_file in data_files:
            data = loadmat(each_file)
            utterance = int(each_file[-8])  # 各speakerの発話番号
            digit = int(each_file[-5]) # 発話された数字
            if utterance in utterance_train_list:  # 訓練用
                # 入力データ（構造体'spec'に格納されている）
                train_input = np.vstack((train_input, data['spec'].T))
                # 出力データ（n_tau x 10，値はすべて'-1'）
                tmp = -np.ones([data['spec'].shape[1], 10])
                tmp[:, digit] = 1  # digitの列のみ1
                train_output = np.vstack((train_output, tmp))
                # データ長
                train_length = np.hstack((train_length, data['spec'].shape[1]))
                # 正解ラベル
                train_label = np.hstack((train_label, digit))
            else:  # 検証用
                # 入力データ（構造体'spec'に格納されている）
                test_input = np.vstack((test_input, data['spec'].T))
                # 出力データ（n_tau x 10，値はすべて'-1'）
                tmp = -np.ones([data['spec'].shape[1], 10])
                tmp[:, digit] = 1  # digitの列のみ1
                test_output = np.vstack((test_output, tmp))
                # データ長
                test_length = np.hstack((test_length, data['spec'].shape[1]))
                # 正解ラベル
                test_label = np.hstack((test_label, digit))
    else:
        print("ディレクトリ %s にファイルが見つかりません．" % (dir_name))
        return
    return train_input, train_output, train_length, train_label, \
           test_input, test_output, test_length, test_label