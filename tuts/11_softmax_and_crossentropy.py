import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)


def cross_entropy(actual, pred):
    loss = -np.sum(actual * np.log(pred))
    return loss # / float(pred.shape[0], divide by N to normalize


# y must be one hot enconded, for 3 classes:
# 0: [1 0 0]
# 1: [0 1 0]
# 2: [0 0 1]
y = np.array([1, 0, 0])

good_y_hat = np.array([0.7, 0.2, 0.1])
bad_y_hat = np.array([0.1, 0.3, 0.6])
lg = cross_entropy(y, good_y_hat)
lb = cross_entropy(y, bad_y_hat)

print(f'Low loss: {lg:.4f}')
print(f'High loss: {lb:.4f}')


print('---Now with pytorch---')
# y does not to be a one hot encoded array
# example with 3 classes
y = torch.tensor([2, 0, 1])
# predictions do not need to be probabilities, no softmax should be applied,
# In other words these below are the logits
# These are good predictions because the max value matches with the positions of
# the true labels,                      2      0                     1
y_good_pred = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y_bad_pred = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

lossp = nn.CrossEntropyLoss()
lg = lossp(y_good_pred, y)
lb = lossp(y_bad_pred, y)

print(f'Low loss: {lg:.4f}')
print(f'High loss: {lb:.4f}')

print(lg.item())
print(lb.item())

print(torch.max(y_good_pred, 1))
print(torch.max(y_bad_pred, 1))
