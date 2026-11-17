import numpy as np

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    return Sigmoid(x)* (1-Sigmoid(x))


X = np.array([[2.0,3.0],[1.0,1.5],
              [0.5,0.2],[-2.0,-3.0],
              [-1.0,-0.5],[-0.2,-0.8],
              [1.5,-0.5],[-1.5,0.5]])

Y = np.array([[1],[1],[1],[0],[0],[0],[1],[0]])

W = np.zeros((2,1))
B = 0

X_TEST = np.array([[3.0,1.0],[-2.5,-1.0],[0.4,-0.1]])

learning_rate = 0.01

for i in range(5000):
    Z = (X @ W) + B
    A = Sigmoid(Z)
    dw = 1/len(X) * np.sum((A - Y) * X)
    db = 1/len(X) * np.sum( A - Y)

    W = W - learning_rate * dw
    B = B - learning_rate * db

    print(f"{A}")

Z = (X_TEST @ W) + B
A = Sigmoid(Z)
print(f"{A}")