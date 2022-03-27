import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# define data path
DATA_PATH = "./HW4_RETAIN-lib/data/"

pids = pickle.load(open(os.path.join(DATA_PATH,'train/pids.pkl'), 'rb'))
vids = pickle.load(open(os.path.join(DATA_PATH,'train/vids.pkl'), 'rb'))
hfs = pickle.load(open(os.path.join(DATA_PATH,'train/hfs.pkl'), 'rb'))
seqs = pickle.load(open(os.path.join(DATA_PATH,'train/seqs.pkl'), 'rb'))
types = pickle.load(open(os.path.join(DATA_PATH,'train/types.pkl'), 'rb'))
rtypes = pickle.load(open(os.path.join(DATA_PATH,'train/rtypes.pkl'), 'rb'))

assert len(pids) == len(vids) == len(hfs) == len(seqs) == 1000
assert len(types) == 619

print("Patient ID:", pids[3])
print("Heart Failure:", hfs[3])
print("# of visits:", len(vids[3]))
for visit in range(len(vids[3])):
    print(f"\t{visit}-th visit id:", vids[3][visit])
    print(f"\t{visit}-th visit diagnosis labels:", seqs[3][visit])
    print(f"\t{visit}-th visit diagnosis codes:", [rtypes[label] for label in seqs[3][visit]])

print("number of heart failure patients:", sum(hfs))
print("ratio of heart failure patients: %.2f" % (sum(hfs) / len(hfs)))

from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, seqs, hfs):
        self.x = seqs
        self.y = hfs
        self.n_samples = len(hfs)

    def __len__(self):

        """
        TODO: Return the number of samples (i.e. patients).
        """

        # your code here
        return self.n_samples

    def __getitem__(self, index):

        """
        TODO: Generates one sample of data.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        """

        # your code here
        return self.x[index], self.y[index]



dataset = CustomDataset(seqs, hfs)

assert len(dataset) == 1000

def collate_fn(data):
    """
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
        sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
        is stored in `mask`.

    Arguments:
        data: a list of samples fetched from `CustomDataset`

    Outputs:
        x: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        rev_x: same as x but in reversed time. This will be used in our RNN model for masking
        rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
        y: a tensor of shape (# patiens) of type torch.float

    Note that you can obtains the list of diagnosis codes and the list of hf labels
        using: `sequences, labels = zip(*data)`
    """

    sequences, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)

    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            """
            TODO: update `x`, `rev_x`, `masks`, and `rev_masks`
            """
            # your code here
            x[i_patient, j_visit, 0:len(visit)] = torch.tensor(visit, dtype=torch.long)
            rev_idx = num_visits[i_patient]-j_visit-1
            rev_x[i_patient, rev_idx, 0:len(visit)] = torch.tensor(visit, dtype=torch.long)
            masks = (x > 0)
            rev_masks = (rev_x > 0)

    return x, masks, rev_x, rev_masks, y

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
loader_iter = iter(loader)
x, masks, rev_x, rev_masks, y = next(loader_iter)

assert x.dtype == rev_x.dtype == torch.long
assert y.dtype == torch.float
assert masks.dtype == rev_masks.dtype == torch.bool

assert x.shape == rev_x.shape == masks.shape == rev_masks.shape == (10, 3, 24)
assert y.shape == (10,)

from torch.utils.data.dataset import random_split

split = int(len(dataset)*0.8)

lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)

print("Length of train dataset:", len(train_dataset))
print("Length of val dataset:", len(val_dataset))


def load_data(train_dataset, val_dataset, collate_fn):

    '''
    TODO: Implement this function to return the data loader for  train and validation dataset.
    Set batchsize to 32. Set `shuffle=True` only for train dataloader.

    Arguments:
        train dataset: train dataset of type `CustomDataset`
        val dataset: validation dataset of type `CustomDataset`
        collate_fn: collate function

    Outputs:
        train_loader, val_loader: train and validation dataloaders

    Note that you need to pass the collate function to the data loader `collate_fn()`.
    '''

    batch_size = 32
    # your code here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, val_loader

train_loader, val_loader = load_data(train_dataset, val_dataset, collate_fn)

assert len(train_loader) == 25, "Length of train_loader should be 25, instead we got %d"%(len(train_loader))


class AlphaAttention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        """
        Define the linear layer `self.a_att` for alpha-attention using `nn.Linear()`;
        
        Arguments:
            hidden_dim: the hidden dimension
        """

        self.a_att = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        """
        TODO: Implement the alpha attention.

        Arguments:
            g: the output tensor from RNN-alpha of shape (batch_size, seq_length, hidden_dim)

        Outputs:
            alpha: the corresponding attention weights of shape (batch_size, seq_length, 1)

        HINT: consider `torch.softmax`
        """

        # your code here
        out = self.a_att(g)
        # softmax on dim 1 or 2 ??
        alphas = torch.softmax(out, dim=1)

        return alphas


class BetaAttention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        """
        Define the linear layer `self.b_att` for beta-attention using `nn.Linear()`;
        
        Arguments:
            hidden_dim: the hidden dimension
        """

        self.b_att = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, h):
        """
        TODO: Implement the beta attention.

        Arguments:
            h: the output tensor from RNN-beta of shape (batch_size, seq_length, hidden_dim)

        Outputs:
            beta: the corresponding attention weights of shape (batch_size, seq_length, hidden_dim)

        HINT: consider `torch.tanh`
        """

        # your code here
        out = self.b_att(h)
        betas = torch.tanh(out)

        return betas

def attention_sum(alpha, beta, rev_v, rev_masks):
    """
    TODO: mask select the hidden states for true visits (not padding visits) and then
        sum the them up.

    Arguments:
        alpha: the alpha attention weights of shape (batch_size, seq_length, 1)
        beta: the beta attention weights of shape (batch_size, seq_length, hidden_dim)
        rev_v: the visit embeddings in reversed time of shape (batch_size, # visits, embedding_dim)
        rev_masks: the padding masks in reversed time of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)

    NOTE: Do NOT use for loop.
    """

    # your code here
    #rev_masks_ = torch.sum(rev_masks, dim=2) > 0
    rev_masks_, _ = torch.max(rev_masks, dim=2)
    rev_masks_ = rev_masks_.unsqueeze(-1)
    # Each visit embedding vector is multiplied by 0 or 1, so embeddings for padded visits are set to 0.
    rev_v_ = rev_v * rev_masks_
    # do we expect hidden_dim and embedding_dim to be same size?
    # they use nn.GRU input_size = 128 and hidden_size = 128
    c_ = alpha * beta * rev_v_
    ## Which dimension should I sum up 1 or 2?
    c = torch.sum(c_, 1)

    # How to use it as masking instead of the product trick...?
    # rev_v_2 = rev_v[rev_masks_, range(128)]
    return c


def sum_embeddings_with_mask(x, masks):
    """
    Mask select the embeddings for true visits (not padding visits) and then sum the embeddings for each visit up.

    Arguments:
        x: the embeddings of diagnosis sequence of shape (batch_size, # visits, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)
    """

    x = x * masks.unsqueeze(-1)
    x = torch.sum(x, dim = -2)
    return x

class RETAIN(nn.Module):

    def __init__(self, num_codes, embedding_dim=128):
        super().__init__()
        # Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        # Define the RNN-alpha using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_a = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the RNN-beta using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_b = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the alpha-attention using `AlphaAttention()`;
        self.att_a = AlphaAttention(embedding_dim)
        # Define the beta-attention using `BetaAttention()`;
        self.att_b = BetaAttention(embedding_dim)
        # Define the linear layers using `nn.Linear()`;
        self.fc = nn.Linear(embedding_dim, 1)
        # Define the final activation layer using `nn.Sigmoid().
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            rev_x: the diagnosis sequence in reversed time of shape (# visits, batch_size, # diagnosis codes)
            rev_masks: the padding masks in reversed time of shape (# visits, batch_size, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        # 1. Pass the reversed sequence through the embedding layer;
        rev_x = self.embedding(rev_x)
        # 2. Sum the reversed embeddings for each diagnosis code up for a visit of a patient.
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        # 3. Pass the reversed embegginds through the RNN-alpha and RNN-beta layer separately;
        g, _ = self.rnn_a(rev_x)
        h, _ = self.rnn_b(rev_x)
        # 4. Obtain the alpha and beta attentions using `AlphaAttention()` and `BetaAttention()`;
        rev_masks_, _ = torch.max(rev_masks, dim=2)
        rev_masks_ = rev_masks_.unsqueeze(-1)
        #alpha = self.att_a(g)
        #beta = self.att_b(h)

        alpha = self.att_a(g * rev_masks_)
        beta = self.att_b(h * rev_masks_)
        # 5. Sum the attention up using `attention_sum()`;
        c = attention_sum(alpha, beta, rev_x, rev_masks)
        # 6. Pass the context vector through the linear and activation layers.
        logits = self.fc(c)
        probs = self.sigmoid(logits)
        return probs.squeeze()


# load the model here
retain = RETAIN(num_codes = len(types))

assert retain.att_a.a_att.in_features == 128, "alpha attention input features is wrong"
assert retain.att_a.a_att.out_features == 1, "alpha attention output features is wrong"
assert retain.att_b.b_att.in_features == 128, "beta attention input features is wrong"
assert retain.att_b.b_att.out_features == 128, "beta attention output features is wrong"

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def eval(model, val_loader):

    """
    Evaluate the model.

    Arguments:
        model: the RNN model
        val_loader: validation dataloader

    Outputs:
        precision: overall precision score
        recall: overall recall score
        f1: overall f1 score
        roc_auc: overall roc_auc score

    REFERENCE: checkout https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """

    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit = model(x, masks, rev_x, rev_masks)
        """
        TODO: obtain the predicted class (0, 1) by comparing y_logit against 0.5, 
              assign the predicted class to y_hat.
        """
        # your code here
        y_hat = y_logit > 0.5
        y_score = torch.cat((y_score,  y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc


def train(model, train_loader, val_loader, n_epochs):
    """
    Train the model.

    Arguments:
        model: the RNN model
        train_loader: training dataloder
        val_loader: validation dataloader
        n_epochs: total number of epochs
    """

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, masks, rev_x, rev_masks)
            """ 
            TODO: calculate the loss using `criterion`, save the output to loss.
            """

            # your code here
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        p, r, f, roc_auc = eval(model, val_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch+1, p, r, f, roc_auc))
    return round(roc_auc, 2)

retain = RETAIN(num_codes = len(types))

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(retain.parameters(), lr=1e-3)

n_epochs = 5
train(retain, train_loader, val_loader, n_epochs)

lr_hyperparameter = [1e-1, 1e-3]
embedding_dim_hyperparameter = [8, 128]
n_epochs = 5
results = {}

for lr in lr_hyperparameter:
    for embedding_dim in embedding_dim_hyperparameter:
        print ('='*50)
        print ({'learning rate': lr, "embedding_dim": embedding_dim})
        print ('-'*50)
        """ 
        TODO: 
            1. Load the model by specifying `embedding_dim` as input to RETAIN. It will create different model with different embedding dimension.
            2. Load the loss function `nn.BCELoss`
            3. Load the optimizer `torch.optim.Adam` with learning rate using `lr` variable
        """
        # your code here
        retain = RETAIN(num_codes = len(types), embedding_dim=embedding_dim)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(retain.parameters(), lr=lr)

        roc_auc = train(retain, train_loader, val_loader, n_epochs)
        results['lr:{},emb:{}'.format(str(lr), str(embedding_dim))] = roc_auc

assert results['lr:0.1,emb:128'] < 0.7, "auc roc should be below 0.7! Since higher learning rate of 0.1 will not allow the model to converge."

print(results)