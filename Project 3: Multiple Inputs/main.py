import numpy as np

X = np.array([[1,1,2],[2,0,-2],[-1,3,4],[0,-2,2],[3,1,0],[-2,-1,-4]])
Y = np.array([[6.0],[9.0],[-3.0],[9.0],[11.0],[-2.0]])

X_TEST = np.array([[1,2,4],[-1,-1,2]])
Y_TEST = np.array([5.0,4.0])

W = np.random.random((3,1))
B = 0
learning_rate = 0.01

def train():
    global W,B
    Y_pred = X @ W + B
    dw = 2/len(X) * X.T @ (Y_pred - Y)
    db = 2/len(X) * np.sum((Y_pred - Y))

    W = W - learning_rate * dw
    B = B - learning_rate * db

for i in range(5000):
    train()

Z = X_TEST @ W + B
print(f"{Z} {Y_TEST}")