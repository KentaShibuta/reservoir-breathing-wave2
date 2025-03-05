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