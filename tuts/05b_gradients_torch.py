import torch
import torch.nn as nn

# y = 2 * x <- the w we are looking is 2
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
X = X.reshape((X.shape[0], 1))
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
Y = Y.reshape((Y.shape[0], 1))
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples={n_samples}, n_features={n_features}')

input_size = n_features
output_size = n_features

# Our model has one layer which consists of a Linear Regression
model = nn.Linear(input_size, output_size)


# Training
learning_rate = 0.1
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_hat = model(X)
    l = loss(Y, y_hat)
    l.backward()

    # This updates the parameters
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0, 0]:.3f}, loss = {l:.8f}')

print(f'Prediction for f(5) = {model(X_test).item():.3f}')



