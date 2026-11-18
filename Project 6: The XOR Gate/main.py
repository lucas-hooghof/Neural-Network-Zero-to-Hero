import numpy as np


X_train = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y_train = np.array([[0],[1],[1],[0]])

W1 = np.random.randn(2,8)
W2 = np.random.randn(8,1)

B1 = np.random.randn(1,8)
B2 = np.random.randn(1,1)

learning_rate = 0.01

for i in range(5000):
    Z1 = X_train @ W1 + B1 
    A1 = np.maximum(0,Z1)

    Z2 = A1 @ W2 + B2
    Y_hat = Z2 

    dZ2 = Y_hat - Y_train
    dW2 = A1.T @ dZ2 / len(X_train)
    dB2 = np.sum(dZ2,axis=0,keepdims=True) / len(X_train)

    dZ1 = (dZ2 @ W2.T) * np.where(Z1 > 0, 1, 0)
    dW1 = X_train.T @ dZ1 / len(X_train)
    dB1 = np.sum(dZ1,axis=0,keepdims=True) / len(X_train)

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2 

    B1 = B1 - learning_rate * dB1
    B2 = B2 - learning_rate * dB2

Z1 = np.maximum(0,(X_train @ W1) + B1)
Z2 = (Z1 @ W2) + B2

print(Z2)

