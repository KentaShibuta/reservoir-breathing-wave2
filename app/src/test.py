#from movie_analyzer import MovieAnalyzer

#movie_file = "/root/app/data/20241227_sophie_1_stabilization.mp4"
#analyzer = MovieAnalyzer(movie_file)
#analyzer.GetColor(show=True)

"""
from mylibs import Point

p = Point(4, 8)
print(p.X(), p.Y(), p.sum)
"""

"""
import numlib
import numpy as np

wout = np.array([[1.2, 3.4, 5.6], [7.8, 9.0, 2.3]], dtype=np.float32)
x = np.array([5, 6, 7], dtype=np.int32)

y = numlib.print_array(wout, x)

print(y)
"""

"""
import numpy as np
from predict import Predict
u = np.array([
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]]
    ], dtype=np.float32)
w_in = np.array([
        [1.2, 3.4], 
        [7.8, 9.0],
        [1.2, 3.4], 
        [7.8, 9.0]
    ], dtype=np.float32)
w = np.array([
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0]
    ], dtype=np.float32)
w_out = np.array([
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
    ], dtype=np.float32)
x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
alpha = 1.0

p = Predict(u, w_in, w, w_out, x, alpha)
p.Print()
y = p.Run()
print(y)
"""

"""
import numpy as np
from esn import ESN
u = np.array([
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]]
    ], dtype=np.float32)
w_in = np.array([
        [1.2, 3.4], 
        [7.8, 9.0],
        [1.2, 3.4], 
        [7.8, 9.0]
    ], dtype=np.float32)
w = np.array([
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0]
    ], dtype=np.float32)
w_out = np.array([
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
    ], dtype=np.float32)
x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
alpha = 1.0


esn = ESN(u, w_in, w, w_out, x, alpha)
esn.Print()
y = esn.Predict()

print(y)

esn.Randnumer_test(4, 3, 1.0)
esn.Generate_erdos_renyi_test(10, 0.5)
"""
"""
y = np.zeros((6, 3), dtype=np.float32)
for i in range(6):
    for j in range(3):
        x_in = np.dot(w_in, u[i][j])
        #print(x_in)
        w_dot_x = np.dot(w, x)
        x_pred = x
        x = (1.0 - 1) * x_pred + 1.0 * np.tanh(w_dot_x + x_in)

        if (j == 3 - 1):
            # yの値を求める
            y_pred = np.dot(w_out, x)
            # y_predをaddする
            y[i] = y_pred
                
print(y)
"""
"""
import numpy as np
from esn import ESN

N_u = 2
N_x = 4
N_y = 3
density=0.5
input_scale=1.0
rho=0.95
leaking_rate=1.0

u = np.array([
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]], 
        [[1.2, 3.4], [7.8, 9.0], [7.8, 9.0]]
    ], dtype=np.float32)

w_out = np.array([
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
        [1.2, 3.4, 5.6, 4.0],
    ], dtype=np.float32)

esn = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate)
esn.SetInput(u, w_out)
esn.Print()

y = esn.Predict()

print(y)
"""

"""
import numpy as np

mat = np.array([
    [0, 0, 0, 1.30172],
    [0, 0, 0.16971, 0],
    [-0, 0.467214, 0, -0.437171], 
    [-0.235993, -0, 1.48135, -0]
], dtype=np.float32)

print(mat)

eigv_list = np.linalg.eig(mat)[0]
sp_radius = np.max(np.abs(eigv_list))
print(sp_radius)
"""

"""
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
x = np.reshape(x, (-1, 1))
print("x:")
print(x)

print("xt:")
print(x.T)

print("x * x.T:")

x_xt = np.dot(x, x.T)

print(x_xt)"
"""
import networkx as nx

def make_connection(N_x, density, rho, seed=0):
        # Erdos-Renyiランダムグラフ
        m = int(N_x*(N_x-1)*density/2)  # 総結合数
        G = nx.gnm_random_graph(N_x, m, seed)

        # 行列への変換(結合構造のみ）
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed=seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W
"""
import numpy as np
N_x = 500
N_u = 60000
N_y = 1
input_scale = 1.0
density = 0.1
rho = 0.9
seed = 0
"""

"""
np.random.seed(seed=seed)
w_in = np.random.uniform(-input_scale, input_scale, (N_x, N_u))
w = make_connection(N_x, density, rho)
w_out = np.random.normal(size=(N_y, N_x))
"""

"""
np.random.default_rng(seed)
w_in = np.random.default_rng.uniform(-input_scale, input_scale, (N_x, N_u))
w = make_connection(N_x, density, rho)
w_out = np.random.default_rng.normal(size=(N_y, N_x))

print("[result] w_in")
print(w_in.shape)
print(w_in)
w_in_sum = 0.0
for i in range(N_x):
    for j in range(N_u):
          w_in_sum += w_in[i,j]

w_in_ave = w_in_sum / (1.0 * N_x * N_u)
print(f"w_in average: {w_in_ave}")

w_in_sum2 = 0.0
for i in range(N_x):
    for j in range(N_u):
          w_in_sum2 += (w_in[i,j] - w_in_ave) * (w_in[i,j] - w_in_ave)

w_in_variance = w_in_sum2 / (1.0 * N_x * N_u)
print(f"w_in variance: {w_in_variance}")

print("[result] w")
print(w.shape)
print(w)

print("[result] w_out")
print(w_out.shape)
print(w_out)"
"""

import numpy as np

# 600x600のNumPy配列を作成し、すべての要素に1497を代入
x = np.full((500, 500), 1497)

#x_inv = np.linalg.pinv(x)
nl = np.linalg
u, s, vT = nl.svd(x, full_matrices=False)
print("VT:")
print(vT)
print("s:")
print(s)
print("u:")
print(u)

eps = 1e-15 # 0とみなす特異値の閾値

# t <= eps なる特異値 t は0にし，そうでないものは逆数1/tをとる
s_inv = np.zeros_like(s)
s_inv[s > eps] = 1.0 / s[s > eps]

# sを2次元ベクトルから2*2の対角行列にする
sigma = np.diag(s_inv)  # `s_inv` を対角成分に持つ対角行列を作成

# 特異値分解の結果から一般逆行列を構成する
x_inv = np.dot(np.dot(vT.T, sigma), u.T)

print("x_inv:")
print(x_inv)


# 条件1: A * A^+ * A = A
result1 = np.dot(x, np.dot(x_inv, x))
condition1 = np.allclose(result1, x)

# 条件2: A^+ * A * A^+ = A^+
result2 = np.dot(x_inv, np.dot(x, x_inv))
condition2 = np.allclose(result2, x_inv)

# 結果を表示
print("result1:")
print(result1)
print("result2:")
print(result2)
print(f"Condition 1 (A * A^+ * A = A): {condition1}")
print(f"Condition 2 (A^+ * A * A^+ = A^+): {condition2}")
