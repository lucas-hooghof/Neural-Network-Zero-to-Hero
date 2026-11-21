import numpy as np


class Adam:

  def __init__(self):
    self.v = 0
    self.m = 0

    self.beta1 = 0.9
    self.beta2 = 0.999
    self.eps = 10**-8

    self.t = 0

  def update(self, g):
    self.t += 1
    self.m = self.beta1 * self.m + (1 - self.beta1) * g
    self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)

    mhat = self.m / (1 - self.beta1**self.t)
    vhat = self.v / (1 - self.beta2**self.t)

    param = mhat / (np.sqrt(vhat) + self.eps)

    return param


def create_spiral_data(samples_per_class=100, classes=3, noise_std=0.2):
  X = np.zeros((samples_per_class * classes, 2))
  y = np.zeros(samples_per_class * classes)

  for c in range(classes):
    ix = range(samples_per_class * c, samples_per_class * (c + 1))
    r = np.arange(samples_per_class) / samples_per_class
    theta = (
        4 * r
        + (2 * np.pi * c) / classes
        + np.random.randn(samples_per_class) * noise_std
    )

    X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
    y[ix] = c

  return X, y


def ReLu(Z):
  return np.maximum(0, Z)


def softmax(Z):
  max_z = np.max(Z, axis=1, keepdims=True)
  exp_Z = np.exp(Z - max_z)
  return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


C = 3

np.random.seed(42)
X, y = create_spiral_data(samples_per_class=100, classes=C, noise_std=0.2)
learning_rate = 0.0001
Y = np.eye(C)[y.ravel().astype(int)]

indices = np.random.permutation(len(X))

X = X[indices]
Y = Y[indices]

W1 = np.random.randn(2, 64)
W2 = np.random.randn(64, 64)
W3 = np.random.randn(64, 3)

B1 = np.zeros((1, 64))
B2 = np.zeros((1, 64))
B3 = np.zeros((1, 3))

W1Adam = Adam()
W2Adam = Adam()
W3Adam = Adam()
B1Adam = Adam()
B2Adam = Adam()
B3Adam = Adam()


def forward(X_input, Y_target):
  Z1 = (X_input @ W1) + B1
  A1 = ReLu(Z1)

  Z2 = (A1 @ W2) + B2
  A2 = ReLu(Z2)

  Y_pred = softmax((A2 @ W3) + B3)

  loss = np.mean((Y_pred - Y_target) ** 2)

  return (loss, Z1, Z2, Y_pred, A1, A2)


for i in range(50000):
  loss, Z1, Z2, Y_pred, A1, A2 = forward(X, Y)

  print(loss)

  dY = Y_pred - Y
  dW3 = A2.T @ dY
  dB3 = np.sum(dY, axis=0, keepdims=True)

  dZ2 = (dY @ W3.T) * (Z2 > 0)
  dW2 = A1.T @ dZ2
  dB2 = np.sum(dZ2, axis=0, keepdims=True)

  dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
  dW1 = X.T @ dZ1
  dB1 = np.sum(dZ1, axis=0, keepdims=True)

  W1 = W1 - learning_rate * W1Adam.update(dW1)
  W2 = W2 - learning_rate * W2Adam.update(dW2)
  W3 = W3 - learning_rate * W3Adam.update(dW3)

  B1 = B1 - learning_rate * B1Adam.update(dB1)
  B2 = B2 - learning_rate * B2Adam.update(dB2)
  B3 = B3 - learning_rate * B3Adam.update(dB3)

# --- Testing ---
np.random.seed(99)
X_test, y_test = create_spiral_data(
    samples_per_class=50, classes=C, noise_std=0.2
)
Y_test = np.eye(C)[y_test.ravel().astype(int)]

ind = np.random.permutation(len(X_test))

X_test = X_test[ind]
Y_test = Y_test[ind]

test_loss, _, _, Y_pred_test, _, _ = forward(X_test, Y_test)

pred_classes = np.argmax(Y_pred_test, axis=1)
true_classes = np.argmax(Y_test, axis=1)
accuracy = np.mean(pred_classes == true_classes)

print("\n--- Test Results ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nFirst 10 Y_pred (probabilities):\n", np.round(Y_pred_test[:10], 3))
print("\nFirst 10 Y_pred (chosen class):\n", pred_classes[:10])
print("First 10 Y_test (true class):  \n", true_classes[:10])