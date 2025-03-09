import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pickle

class dataset:
    def __init__(self, index, explanatory, response):
        self.index = index
        self.explanatory = explanatory
        self.response = response
    
    def getdata(self):
        return self.index, self.explanatory, self.response
    
    def getsize(self):
        return self.index.shape, self.explanatory.shape, self.response.shape
    
    def show(self):
        print(f"inputdata_indexのサイズ: {self.index.shape}")
        print(f"inputdata_indexのデータ型: {self.index[0].dtype}")
        print(f"explanatoryのサイズ: {self.explanatory.shape}")
        print(f"explanatoryのデータ型: {self.explanatory[0,0].dtype}")
        print(f"responseのサイズ: {self.response.shape}")
        print(f"responseのデータ型: {self.response[0].dtype}")


class data_splitter:
    def __init__(self, original_data, test_size, isTrain = True):
        self.window_size = 5
        self.N = 1

        explanatory = original_data[:, :-1].astype(np.uint8)
        response = original_data[:, -1].astype(np.float32)
        
        # インデックスデータの作成
        index = np.array(range(original_data.shape[0])).astype(np.uint32)

        if isTrain == True:
            # 訓練データとテストデータで分割
            shuffle = False
            train_index, test_index = train_test_split(index, test_size=test_size, shuffle=shuffle)
            train_explanatory, test_explanatory = train_test_split(explanatory, test_size=test_size, shuffle=shuffle)
            train_response, test_response = train_test_split(response, test_size=test_size, shuffle=shuffle)

            del(original_data)
            del(explanatory)
            del(response)
            del(index)

            self.train = dataset(train_index, train_explanatory, train_response)
            self.test = dataset(test_index, test_explanatory, test_response)

            del(train_index)
            del(test_index)
            del(train_explanatory)
            del(test_explanatory)
            del(train_response)
            del(test_response)

            print("train show")
            self.train.show()
            print("test show")
            self.test.show()
        else:
            self.input = dataset(index, explanatory, response)

            del(original_data)
            del(explanatory)
            del(response)
            del(index)

    def create_batch(self, show=False, isTrain=True):
        print("create batch")

        if isTrain == True:
            # 学習用データ作成
            train_index, train_explanatory, train_response = self.train.getdata()
            train_index_size, train_explanatory_size, train_response_size = self.train.getsize()
            del(self.train)

            self.series_size = train_explanatory_size[1] # 説明変数の系列数
            n_train = train_explanatory_size[0] - self.window_size + 1 - self.N

            print("create train data")
            # 正解データを準備
            train = np.zeros((n_train, self.window_size, self.series_size))
            train_labels = np.zeros(n_train)
            train_labels_index = np.zeros(n_train)
            for i in range(n_train):
                #print(i)
                response_value_index = i + (self.window_size - 1) + self.N
                train[i] = train_explanatory[i:i+self.window_size]
                train_labels[i] = train_response[response_value_index]
                train_labels_index[i] = train_index[response_value_index]
            print("end create train data")

            del(train_index)
            del(train_explanatory)
            del(train_response)

            test_index, test_explanatory, test_response = self.test.getdata()
            test_index_size, test_explanatory_size, test_response_size = self.test.getsize()
            del(self.test)
            n_test = test_explanatory_size[0] - self.window_size + 1 - self.N

            print("create test data")
            # テストデータを準備
            test = np.zeros((n_test, self.window_size, self.series_size))
            test_labels = np.zeros(n_test)
            test_labels_index = np.zeros(n_test)
            for i in range(n_test):
                response_value_index = i + (self.window_size - 1) + self.N
                test[i] = test_explanatory[i:i+self.window_size]
                test_labels[i] = test_response[response_value_index]
                test_labels_index[i] = test_index[response_value_index]
            print("end create test data")

            del(test_index)
            del(test_explanatory)
            del(test_response)

            if show == True:
                plt.plot(train_labels_index, train_labels, label='train')
                plt.plot(test_labels_index, test_labels, label='test')
                plt.legend()

                plt.xlabel("time step")
                plt.ylabel("Y")
                plt.show()

            print(f"train_size:{train.shape}")
            print(f"train_labels_size:{train_labels.shape}")
            print(f"test_size:{test.shape}")
            print(f"test_labels_size:{test_labels.shape}")

            return train, train_labels, train_labels_index, test, test_labels, test_labels_index
        else:
            # 学習用データ作成
            index, explanatory, response = self.input.getdata()
            index_size, explanatory_size, response_size = self.input.getsize()
            del(self.input)

            self.series_size = explanatory_size[1] # 説明変数の系列数
            n_input = explanatory_size[0] - self.window_size + 1 - self.N

            print("create train data")
            # 正解データを準備
            input = np.zeros((n_input, self.window_size, self.series_size))
            #input_labels = np.zeros(n_input)
            #input_labels_index = np.zeros(n_input)
            for i in range(n_input):
                #print(i)
                response_value_index = i + (self.window_size - 1) + self.N
                input[i] = explanatory[i:i+self.window_size]
                #input_labels[i] = response[response_value_index]
                #input_labels_index[i] = index[response_value_index]
            print("end create train data")

            del(index)
            del(explanatory)
            del(response)

            return input#, input_labels, input_labels_index

    def read_binary(self, filename):
        with open(filename, mode='br') as fi:
            return pickle.load(fi)