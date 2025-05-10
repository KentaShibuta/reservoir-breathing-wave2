import numpy as np

np.random.seed(seed=0)

# NARMAモデル
class NARMA:
    # パラメータの設定
    def __init__(self, m, a1, a2, a3, a4):
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def generate_data(self, T, y_init, seed=0):
        n = self.m
        y = y_init
        np.random.seed(seed=seed)
        u = np.random.uniform(0, 0.5, T)

        # 時系列生成
        while n < T:
            y_n = self.a1*y[n-1] + self.a2*y[n-1]*(np.sum(y[n-self.m:n-1])) \
                + self.a3*u[n-self.m]*u[n] + self.a4
            y.append(y_n)
            n += 1

        return u, np.array(y)