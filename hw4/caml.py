import os
import csv
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# Define data path
DATA_PATH = "./HW4_CAML-lib/data/"


def to_index(sequence, token2idx):
    """
    TODO: convert the sequnce of tokens to indices.
    If the word in unknown, then map it to '<unk>'.

    INPUT:
        sequence (type: list of str): a sequence of tokens
        token2idx (type: dict): a dictionary mapping token to the corresponding index

    OUTPUT:
        indices (type: list of int): a sequence of indicies

    EXAMPLE:
        >>> sequence = ['hello', 'world', 'unknown_word']
        >>> token2idx = {'hello': 0, 'world': 1, '<unk>': 2}
        >>> to_index(sequence, token2idx)
        [0, 1, 2]
    """
    # your code here
    return [token2idx.get(token, token2idx['<unk>']) for token in sequence]


sequence = ['hello', 'world', 'unknown_word']
token2idx = {'hello': 0, 'world': 1, '<unk>': 2}
assert to_index(sequence, token2idx) == [0, 1, 2], "to_index() is wrong!"

from torch.utils.data import Dataset

NUM_WORDS = 1253
NUM_CLASSES = 20


class CustomDataset(Dataset):

    def __init__(self, filename):
        # read in the data files
        self.data = pd.read_csv(filename)
        # load word lookup
        self.idx2word, self.word2idx = self.load_lookup(f'{DATA_PATH}/vocab.csv', padding=True)
        assert len(self.idx2word) == len(self.word2idx) == NUM_WORDS

    def load_lookup(self, filename, padding=False):
        """ load lookup for word """
        idx2token = {}
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                idx2token[i] = line
        token2idx = {w:i for i,w in idx2token.items()}
        return idx2token, token2idx

    def __len__(self):

        """
        TODO: Return the number of samples (i.e. admissions).
        """

        # your code here
        return self.data.shape[0]

    def __getitem__(self, index):

        """
        TODO: Generate one sample of data.

        Hint: convert text to indices using to_index();
        """
        data = self.data.iloc[index]
        text = data['Text'].split(' ')
        label = data['Label']
        # convert label string to list
        label = [int(l) for l in label.strip('[]').split(', ')]
        assert len(label) == NUM_CLASSES
        # your code here
        text = to_index(text, self.word2idx)
        # return text as long tensor, labels as float tensor;
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.float)


dataset = CustomDataset(f'{DATA_PATH}/train_df.csv')
assert len(dataset) == 3141, "__len__() is wrong!"

text, labels = dataset[1]

assert type(text) is torch.Tensor, "__getitem__(): text is not tensor!"
assert type(labels) is torch.Tensor, "__getitem__(): labels is not tensor!"
assert text.dtype is torch.int64, "__getitem__(): text is not of type long!"
assert labels.dtype is torch.float32, "__getitem__(): labels is not of type float!"


from torch.nn.utils.rnn import pad_sequence


def collate_fn(data):
    """
    TODO: implement the collate function.

    STEP: 1. pad the text using pad_sequence(). Set `batch_first=True`.
          2. stack the labels using torch.stack().

    OUTPUT:
        text: the padded text, shape: (batch size, max length)
        labels: the stacked labels, shape: (batch size, num classes)
    """
    text, labels = zip(*data)

    # your code here
    text_ = pad_sequence(text, batch_first=True)
    labels_ = torch.stack(labels, dim=0)

    return text_, labels_


from torch.utils.data import DataLoader

dataset = CustomDataset(f'{DATA_PATH}/train_df.csv')
loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
loader_iter = iter(loader)
text, labels = next(loader_iter)

assert text.shape == (10, 104), "collate_fn(): text has incorrect shape!"
assert labels.shape == (10, 20), "collate_fn(): labels has incorrect shape!"

train_set = CustomDataset(f'{DATA_PATH}/train_df.csv')
test_set = CustomDataset(f'{DATA_PATH}/test_df.csv')
train_loader = DataLoader(train_set, batch_size=32, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)

from math import floor
from torch.nn.init import xavier_uniform_


class CAML(nn.Module):

    def __init__(self, kernel_size=10, num_filter_maps=16, embed_size=100, dropout=0.5):
        super(CAML, self).__init__()

        # embedding layer
        self.embed = nn.Embedding(NUM_WORDS, embed_size, padding_idx=0)
        self.embed_drop = nn.Dropout(p=dropout)

        # initialize conv layer as in section 2.1
        self.conv = nn.Conv1d(embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform_(self.conv.weight)

        # context vectors for computing attention as in section 2.2
        self.U = nn.Linear(num_filter_maps, 20)
        xavier_uniform_(self.U.weight)

        # final layer: create a matrix to use for the NUM_CLASSES binary classifiers as in section 2.3
        self.final = nn.Linear(num_filter_maps, NUM_CLASSES)
        xavier_uniform_(self.final.weight)

    def forward_embed(self, text):
        """
        TODO: Feed text through the embedding (self.embed) and dropout layer (self.embed_drop).

        INPUT:
            text: (batch size, seq_len)

        OURPUT:
            text: (batch size, seq_len, embed_size)
        """
        # your code here
        out = self.embed(text)
        out = self.embed_drop(out)

        return out

    def forward_conv(self, text):
        """
        TODO: Feed text through the convolution layer (self.conv) and tanh activation function (torch.tanh)
        in eq (1) in the paper.

        INTPUT:
            text: (batch size, embed_size, seq_len)

        OUTPUT:
            text: (batch size, num_filter_maps, seq_len)
        """
        # your code here
        out = self.conv(text)
        h = torch.tanh(out)

        return h

    def forward_calc_atten(self, text):
        """
        TODO: calculate the attention weights in eq (2) in the paper. Be sure to read the documentation for
        F.softmax()

        INPUT:
            text: (batch size, seq_len, num_filter_maps)

        OUTPUT:
            alpha: (batch size, num_class, seq_len), the attention weights

        STEP: 1. multiply `self.U.weight` with `text` using torch.matmul();
              2. apply softmax using `F.softmax()`.
        """
        # (batch size, seq_len, num_filter_maps) -> (batch size, num_filter_maps, seq_len)
        text = text.transpose(1,2)
        # your code here
        alpha = torch.matmul(self.U.weight, text)
        # which is the dim for softmax?
        alpha_sm = F.softmax(alpha, dim=2)

        return alpha_sm

    def forward_aply_atten(self, alpha, text):
        """
        TODO: apply the attention in eq (3) in the paper.

        INPUT:
            text: (batch size, seq_len, num_filter_maps)
            alpha: (batch size, num_class, seq_len), the attention weights

        OUTPUT:
            v: (batch size, num_class, num_filter_maps), vector representations for each label

        STEP: multiply `alpha` with `text` using torch.matmul().
        """
        # your code here
        vrep = torch.matmul(alpha, text)

        return vrep

    def forward_linear(self, v):
        """
        TODO: apply the final linear classification in eq (5) in the paper.

        INPUT:
            v: (batch size, num_class, num_filter_maps), vector representations for each label

        OUTPUT:
            y_hat: (batch size, num_class), label probability

        STEP: 1. multiply `self.final.weight` v `text` element-wise using torch.mul();
              2. sum the result over dim 2 (i.e. num_filter_maps);
              3. add the result with `self.final.bias`;
              4. apply sigmoid with torch.sigmoid().
        """
        # your code here
        y_hat_1 = torch.mul(self.final.weight, v)
        y_hat_2 = torch.sum(y_hat_1, dim=2)
        y_hat_3 = y_hat_2 + self.final.bias

        y_hat = torch.sigmoid(y_hat_3)

        return y_hat

    def forward(self, text):
        """ 1. get embeddings and apply dropout """
        text = self.forward_embed(text)
        # (batch size, seq_len, embed_size) -> (batch size, embed_size, seq_len);
        text = text.transpose(1, 2)

        """ 2. apply convolution and nonlinearity (tanh) """
        text = self.forward_conv(text)
        # (batch size, num_filter_maps, seq_len) -> (batch size, seq_len, num_filter_maps);
        text = text.transpose(1,2)

        """ 3. calculate attention """
        alpha = self.forward_calc_atten(text)

        """ 3. apply attention """
        v = self.forward_aply_atten(alpha, text)

        """ 4. final layer classification """
        y_hat = self.forward_linear(v)

        return y_hat


model = CAML()

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

from sklearn.metrics import precision_recall_fscore_support


def eval(model, test_loader):

    """
    INPUT:
        model: the CAML model
        test_loader: dataloader

    OUTPUT:
        precision: overall micro precision score
        recall: overall micro recall score
        f1: overall micro f1 score

    REFERENCE: checkout https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """

    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    model.eval()
    for sequences, labels in test_loader:
        """
        TODO: 1. preform forward pass
              2. obtain the predicted class (0, 1) by comparing forward pass output against 0.5, 
                 assign the predicted class to y_hat.
        """
        # your code here
        y_logit = model(sequences)
        y_hat = y_logit > 0.5
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, labels.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return p, r, f


def train(model, train_loader, test_loader, n_epochs):
    """
    INPUT:
        model: the CAML model
        train_loader: dataloder
        val_loader: dataloader
        n_epochs: total number of epochs
    """
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            """ 
            TODO: 1. perform forward pass using `model`, save the output to y_hat;
                  2. calculate the loss using `criterion`, save the output to loss.
            """
            y_hat, loss = None, None
            # your code here
            y_hat = model(sequences)
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        p, r, f = eval(model, test_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}'.format(epoch+1, p, r, f))


# number of epochs to train the model
n_epochs = 20

train(model, train_loader, test_loader, n_epochs)

p, r, f = eval(model, test_loader)
assert f > 0.70, "f1 below 0.70!"