import numpy as np

np.random.seed(42)

X = np.random.randn(3, 4)
Y = np.random.randn(3, 2)

W1 = np.random.randn(4, 8)
B1 = np.zeros((1, 8))
W2 = np.random.randn(8, 2)
B2 = np.zeros((1, 2))


def forward_loss(X, Y, W1, B1, W2, B2):
    Z1 = X @ W1 + B1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + B2
    loss = np.mean((Z2 - Y) ** 2)
    return loss, (Z1, A1, Z2)


loss, (Z1, A1, Z2) = forward_loss(X, Y, W1, B1, W2, B2)

dZ2 = 2 * (Z2 - Y) / Y.size
dW2 = A1.T @ dZ2
dB2 = np.sum(dZ2, axis=0, keepdims=True)

dZ1 = (dZ2 @ W2.T) * np.where(Z1 > 0, 1, 0)
dW1 = X.T @ dZ1
dB1 = np.sum(dZ1, axis=0, keepdims=True)


eps = 1e-5
num_dW1 = np.zeros_like(W1)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        orig_val = W1[i, j]

        W1[i, j] = orig_val + eps
        loss_plus, _ = forward_loss(X, Y, W1, B1, W2, B2)

        W1[i, j] = orig_val - eps
        loss_minus, _ = forward_loss(X, Y, W1, B1, W2, B2)

        num_dW1[i, j] = (loss_plus - loss_minus) / (2 * eps)
        W1[i, j] = orig_val 

rel_error = np.linalg.norm(dW1 - num_dW1) / (
    np.linalg.norm(dW1) + np.linalg.norm(num_dW1)
)
print(f"Relative Error for W1: {rel_error:.12e}")