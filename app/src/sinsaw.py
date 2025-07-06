import numpy as np

# 正弦波とのこぎり波の混合波形生成
class SINSAW:
    def __init__(self, period):
        self.period = period  # 周期

    # 正弦波
    def sinusoidal(self):
        n = np.arange(self.period)
        x = np.sin(2*np.pi*n/self.period)

        return x

    # のこぎり波
    def saw_tooth(self):
        n = np.arange(self.period)
        x = 2*(n/self.period - np.floor(n/self.period+0.5))

        return x

    def make_output(self, label):
        y = np.zeros((self.period, 2))
        y[:, label] = 1

        return y

    # 混合波形及びラベルの出力
    def generate_data(self, label):
        '''
        :param label: 0または1を要素に持つリスト
        :return: u: 混合波形
        :return: d: 2次元ラベル（正弦波[1,0], のこぎり波[0,1]）
        '''
        u = np.empty(0)
        d = np.empty((0, 2))
        for i in label:
            if i:
                u = np.hstack((u, self.saw_tooth()))
            else:
                u = np.hstack((u, self.sinusoidal()))
            d = np.vstack((d, self.make_output(i)))

        return u, d