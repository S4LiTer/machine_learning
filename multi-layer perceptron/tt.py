import numpy as np


x = (np.random.random((5, 5))-0.5)*10
m = np.maximum(x,0)

print(x)
_x = x.copy()
print(m)
x[x>0] = 1
x[x<0] = 0
print(x)
print(_x)