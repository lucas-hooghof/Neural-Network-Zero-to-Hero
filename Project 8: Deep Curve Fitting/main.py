import numpy as np

def ReLu(Z):
    return np.maximum(0,Z)

X = np.linspace(-3, 3, 256).reshape(-1,1)
Y = X**2 - 2*X + 1

W1 = np.random.randn(1, 32) * 0.1
W2 = np.random.randn(32, 32) * 0.1
W3 = np.random.randn(32, 1) * 0.1

B1 = np.random.randn(1,32)
B2 = np.random.randn(1,32)
B3 = np.random.randn(1,1)


learning_rate = 0.005

XTest = np.array([[-2],[0],[1],[3]])

def forward():
    Z1 = (X @ W1) + B1
    A1 = ReLu(Z1)

    Z2 = (A1 @ W2) + B2
    A2 = ReLu(Z2)

    Y_pred = (A2 @ W3) + B3

    loss = np.mean((Y_pred - Y) ** 2)

    return (loss,Z1,Z2,Y_pred,A1,A2)



for i in range(50000):
    loss,Z1,Z2,Y_pred,A1,A2 = forward()

    print(loss)

    dY = 2/len(X) * (Y_pred - Y)

    dW3 = A2.T @ dY
    dB3 = np.sum(dY, axis=0, keepdims=True)

    dZ2 = (dY @ W3.T) * (Z2 > 0)
    dW2 = A1.T @ dZ2
    dB2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2 
    W3 = W3 - learning_rate * dW3 

    B1 = B1 - learning_rate * dB1
    B2 = B2 - learning_rate * dB2
    B3 = B3 - learning_rate * dB3


A1_test = ReLu((XTest @ W1) + B1)
A2_test = ReLu((A1_test @ W2) + B2)
Z = (A2_test @ W3) + B3

print(Z)
    

