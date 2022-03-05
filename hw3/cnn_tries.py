import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score

# record start time
_START_RUNTIME = time.time()

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# Define data and weight path
DATA_PATH = "HW3_CNN-lib/data"


def load_data(data_path=DATA_PATH):

    data_transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])

    train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), data_transformer)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                               shuffle=True)

    val_set = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transformer)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32,
                                             shuffle=False)

    return train_loader, val_loader


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # your code here
        # (224 - 3 + (2*0) / 1)  +  1 = 222
        self.conv1 = nn.Conv2d(3, 6, 3)
        # (222 - 3 + (2*0) / 1)  +  1 = 220
        self.conv2 = nn.Conv2d(6, 8, 3)
        # (220 - 5 + (2*0) / 1)  +  1 = 216
        self.conv3 = nn.Conv2d(8, 10, 5)
        # 216 / 2 = 108
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout()
        self.linear_size = 10*108*108

        fc1_out = int(self.linear_size * 1)
        self.fc1 = nn.Linear(self.linear_size, 2)

        #fc2_out = int(fc1_out * 0.6)
        #self.fc2 = nn.Linear(fc1_out, fc2_out)

        #fc3_out = int(fc2_out * 0.4)
        #self.fc3 = nn.Linear(fc2_out, fc3_out)

        #self.fc4 = nn.Linear(fc3_out, 2)

    def forward(self, x):
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        # your code here
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        #x = self.pool(x)

        x = x.view(-1, self.linear_size)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        x = self.fc1(x)

        #x = torch.sigmoid(x)

        return x


class SimpleCNN2(nn.Module):

    def __init__(self):

        super(SimpleCNN2, self).__init__()
        # your code here
        # (224 - 3 + (2*0) / 1)  +  1 = 222
        self.conv1 = nn.Conv2d(3, 6, 3)
        # (222 - 3 + (2*0) / 1)  +  1 = 220
        self.conv2 = nn.Conv2d(6, 12, 3)
        # 220 / 2 = 110
        self.pool1 = nn.MaxPool2d(2, 2)
        # (110 - 3 + (2*0) / 1)  +  1 = 108
        self.conv3 = nn.Conv2d(12, 24, 3)
        # (108 - 3 + (2*0) / 1)  +  1 = 106
        self.conv4 = nn.Conv2d(24, 36, 3)
        # 106 / 2 = 53
        self.pool2 = nn.MaxPool2d(2, 2)
        # Here we flatten the output of previous layers
        self.dropout = nn.Dropout()
        self.linear_size = 36*53*53

        fc1_out = int(self.linear_size * 0.8)
        self.fc1 = nn.Linear(self.linear_size, fc1_out)

        fc2_out = int(fc1_out * 0.6)
        self.fc2 = nn.Linear(fc1_out, fc2_out)

        fc3_out = int(fc2_out * 0.4)
        self.fc3 = nn.Linear(fc2_out, fc3_out)

        self.fc4 = nn.Linear(fc3_out, 2)

    def forward(self, x):
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        # your code here
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = x.view(-1, self.linear_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        x = self.fc4(x)

        return x


def train_model(model, train_dataloader, n_epoch, optimizer, criterion):
    import torch.optim as optim
    """
    :param model: A CNN model
    :param train_dataloader: the DataLoader of the training data
    :param n_epoch: number of epochs to train
    :return:
        model: trained model
    """
    model.train() # prep model for training

    n_total_steps = len(train_dataloader)
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for i, (data, target) in enumerate(train_dataloader):
            """
            TODO: Within the loop, do the normal training procedures:
                   pass the input through the model
                   pass the output through loss_func to compute the loss (name the variable as *loss*)
                   zero out currently accumulated gradient, use loss.basckward to backprop the gradients, then call optimizer.step
            """
            # your code here
            outputs = model(data)
            loss = criterion(outputs, target) #y.view(y.shape[0], 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_epoch_loss.append(loss.cpu().data.numpy())

            #if (i+1) % 10 == 0:
            print(f'epoch {epoch+1} / {n_epoch}, step {i+1}/ {n_total_steps}, loss = {loss.item():.4f}')
        print(f"Epoch {epoch+1}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model


def eval_model(model, dataloader):

    model.eval()
    Y_pred = []
    Y_test = []
    n_total_steps = len(dataloader)
    for i, (data, target) in enumerate(dataloader):
        # your code here
        preds = model(data)
        _, predicted = torch.max(preds, 1)

        Y_pred.append(predicted.detach().numpy())
        Y_test.append(target.detach().numpy())

        print(f'step {i+1}/ {n_total_steps}')

    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    return Y_pred, Y_test


def get_resnet18():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # We also add a new layer to the model
    model.fc = nn.Linear(num_ftrs, 2)
    return model



train_loader, val_loader = load_data()

model = SimpleCNN()
#model = get_resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 10
model = train_model(model, train_loader, n_epochs, optimizer, criterion)


print('-'*20)
print('Evaluating model...')
y_pred, y_true = eval_model(model, val_loader)
acc = accuracy_score(y_true, y_pred)
print(("Validation Accuracy: " + str(acc)))