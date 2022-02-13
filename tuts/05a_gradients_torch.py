import torch

# y = 2 * x <- the w we are looking is 2
X = torch.tensor([1, 2, 3, 4])
Y = torch.tensor([2, 4, 6, 8])

w = torch.tensor(0.0, requires_grad=True)


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_hat):
    return ((y_hat - y)**2).mean()


# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_hat = forward(X)
    l = loss(Y, y_hat)
    l.backward()

    # this should not be part of the gradient graph tracking
    with torch.no_grad():
        w -= learning_rate * w.grad
        #w.copy_(w_)
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')



