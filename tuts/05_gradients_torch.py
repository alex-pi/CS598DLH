import torch
import numpy as np

# y = 2 * x <- the w we are looking is 2
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_hat):
    return ((y_hat - y)**2).mean()


# gradient
# All the transformations done to x are the forward pass
# MSE = 1/N * (w*x - y)**2
# So we need the derivative of all that
# dJ/Dw = 1/N 2*x (w*x - y)
def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat - y).mean()

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    y_hat = forward(X)
    l = loss(Y, y_hat)
    dw = gradient(X, Y, y_hat)

    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')



