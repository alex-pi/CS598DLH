import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''dataset = torchvision.datasets.MNIST(
    root='./data',
    transform=torchvision.transforms.ToTensor()
)'''

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        item = self.x[index], self.y[index]

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return self.n_samples

class ToTensor():
    def __call__(self, item):
        inputs, targets = item

        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, item):
        inputs, targets = item

        inputs *= self.factor
        return inputs, targets



dataset = WineDataset(transform=ToTensor())

features, labels = dataset[0]
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)

features, labels = dataset[0]
print(features)
print(type(features), type(labels))
