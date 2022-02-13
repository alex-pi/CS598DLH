import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size = 4

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              transform=transform, download=True)

test_data_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

# input channels, out channels, kernel size (filter)
conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# input channels, out channels, kernel size (filter)
conv2 = nn.Conv2d(6, 16, 5)

print(f'Original (batch size, channels, width, height): {images.shape}')
x = conv1(images)
# (32 - 5 + (2*0) / 1)  +  1 = 28
print(f'conv1: {x.shape}')
x = pool(x)
# The 2, 2 pooling reduces the image by a factor of 2
print(f'pool1: {x.shape}')
x = conv2(x)
print(f'conv2: {x.shape}')
x = pool(x)
# The 2, 2 pooling reduces the image by a factor of 2
print(f'pool2: {x.shape}')

