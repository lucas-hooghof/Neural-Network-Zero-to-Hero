import numpy as np

X = np.array([-2,-1,0,1,2,3])
Y = np.array([-3,-1,1,3,5,7])

X_TEST = np.array([4.0,-5.0,0.5])
Y_TEST = np.array([9.0,-9.0,2.0])

W = 0
B = 0

learning_rate = 0.01

for i in range(50000):
    dw = 2/len(X) * np.sum(((X * W + B) - Y) * X)
    db = 2/len(X) * np.sum(((X * W + B) - Y))
    W = W - learning_rate * dw
    B = B - learning_rate * db

Z = W * X_TEST + B
print(f"{Y_TEST} {Z}")