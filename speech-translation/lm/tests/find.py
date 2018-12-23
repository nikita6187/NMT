import numpy as np


a = np.loadtxt("./lm.w_re0")
print(a)
"""
print(np.where(a == 0.00667304))
print(np.where(0.00667305 >= a >= 0.00667304))
"""
print(a.shape)
for x in range(a.shape[0]):
    for y in range(a.shape[1]):
        if -0.01245036 <= a[x, y] <= -0.01245036:
            print(str(x) + " " + str(y) + " " + str(a[x, y]))

