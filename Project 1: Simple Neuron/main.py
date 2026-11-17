import numpy as np

X = np.array([1,2,3,4])
Y = np.array([2,4,6,8])

X_TEST = np.array([5.0,-3.0,10.0])
Y_TEST = np.array([10.0,-6.0,20.0])

W = 0

learning_rate = 0.01

for i in range(5000):
    dw = 2/len(X) * np.sum(((X * W) - Y) * X)
    W = W - learning_rate * dw

Z = W * X_TEST
print(f"{Y_TEST} {Z}")