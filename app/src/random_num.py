import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)
#print(np.random.uniform(-1,1,(3, 3)))

#rs = np.random.RandomState(0)
#print(rs.uniform(-1,1,(3,3)))
#print(rs.randint(0, 2**32, (3,3)))

"""
rs2 = np.random.RandomState(0)
arr = np.empty((3, 3))
for i in range(3):
    for j in range(3):
        arr[i, j] = rs2.uniform(-1, 1)
print(arr)
"""

rs3 = np.random.RandomState(0)
N_u = 60000
N_x = 500
rand_num = N_x * N_u

#arr = np.empty(rand_num)
#for i in range(len(arr)):
#    arr[i] = rs3.uniform(-1, 1)
arr = rs3.uniform(-1, 1, rand_num)

print(f"max:{arr.max()}")
print(f"min:{arr.min()}")
arr = np.sort(arr)
arr = arr.reshape(-1, 1)
print(f"max:{arr.max()}")
print(f"min:{arr.min()}")
#print(arr)
np.savetxt('./random_py.csv', arr)

plt.plot(range(rand_num), arr, label='rand_num')
plt.legend()
plt.xlabel("sample num")
plt.ylabel("rand num")
plt.show()