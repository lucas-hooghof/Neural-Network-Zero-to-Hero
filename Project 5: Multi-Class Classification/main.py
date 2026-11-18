import numpy as np

def softmax(Z):
    max_z = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - max_z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy_loss(Y_hat, Y_true, eps=1e-15):
    return -np.sum(Y_true * np.log(Y_hat + eps)) / Y_hat.shape[0]

np.random.seed(42)
X_train = np.array([
    [ 2.0,  2.0],  
    [ 1.5,  3.0], 
    [ 3.0,  1.0],  
    [-2.0,  2.0],
    [-3.0,  1.5],  
    [-1.5,  2.5], 
    [-2.0, -2.0], 
    [-1.0, -3.0], 
    [-3.0, -1.0]   
])

y_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


X_test = np.array([
    [ 2.5,  2.5],  
    [-2.5,  2.0], 
    [-2.5, -2.5]    
])

C = 3
D = 2
N = 9

W = np.random.randn(D, C) * 0.01
B = np.zeros((1,C))


Y_train = np.zeros((N, C))
Y_train[np.arange(N), y_labels] = 1
learning_rate = 0.01

for i in range(5000):
    Z = (X_train @ W) + B
    Y_hat = softmax(Z)

    dZ = Y_hat - Y_train
    dW = (X_train.T @ dZ) / N
    db = np.sum(dZ,axis=0,keepdims=True) / N

    W = W - learning_rate * dW
    B = B - learning_rate * db


Z_test = np.dot(X_test, W) + B
Y_test_hat = softmax(Z_test)

predictions = np.argmax(Y_test_hat, axis=1)
max_probabilities = np.max(Y_test_hat, axis=1)
print("--- Test Results ---")
for i in range(len(X_test)):
    print(f"Sample {X_test[i]} -> Predicted Class: {predictions[i]} "
          f"| Confidence: {max_probabilities[i]:.4f}")