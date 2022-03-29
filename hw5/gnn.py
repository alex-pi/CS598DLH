import os
import math
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.utils as tg_utils
import torch_geometric

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

DATA_PATH = "./HW5_GNN-lib/data/"

print(torch.__version__)
print(torch_geometric.__version__)

from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
print("num_nodes:", data.num_nodes)
print("num_edges:", data.num_edges)
print("num_node_features:", data.num_node_features)

data = None

# your code here
edge_index = torch.tensor([[0, 0, 2, 1],
                           [1, 2, 1, 3]], dtype=torch.long)

data = Data(x=0, edge_index=edge_index, y=1)

assert data.num_nodes == 4
assert data.num_edges == 4
assert data.y == 1
assert data.num_node_features == 0
assert data.num_edge_features == 0

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root=DATA_PATH, name='MUTAG')
print("len:", len(dataset))
print("num_classes:", dataset.num_classes)
print("num_node_features:", dataset.num_node_features)

data = dataset[0]
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_node_features}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

def graph_stat(dataset):
    """
    TODO: calculate the statistics of the ENZYMES dataset.

    Outputs:
        min_num_nodes: min number of nodes
        max_num_nodes: max number of nodes
        mean_num_nodes: average number of nodes
        min_num_edges: min number of edges
        max_num_edges: max number of edges
        mean_num_edges: average number of edges
    """

    # your code here
    num_nodes_per_graph = []
    num_edges_per_graph = []
    edge_idx_sizes = []

    for g in dataset:
        num_nodes_per_graph.append(g.num_nodes)
        num_edges_per_graph.append(g.num_edges)
        edge_idx_sizes.append(g.edge_index.shape[1])

    num_nodes = np.array(num_nodes_per_graph)
    num_edges = np.array(num_edges_per_graph)
    edge_idx_sizes = np.array(edge_idx_sizes)

    return num_nodes.min(), num_nodes.max(), num_nodes.mean(), num_edges.min(), num_edges.max(), num_edges.mean()


graph_stat(dataset)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

loader_iter = iter(loader)
batch = next(loader_iter)
print(batch)
print(batch.num_graphs)

# shuffle
dataset = dataset.shuffle()
# split
split_idx = int(len(dataset) * 0.8)
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]

print("len train:", len(train_dataset))
print("len test:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        # Initialize the parameters.
        stdv = 1. / math.sqrt(out_channels)
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        """
        TODO:
            1. Generate the adjacency matrix with self-loop \hat{A} using edge_index.
            2. Calculate the diagonal degree matrix \hat{D}.
            3. Calculate the output X' with torch.mm using the equation above.
        """
        # your code here
        n_nodes = x.shape[0]
        #A = tg_utils.to_dense_adj(edge_index).squeeze(0)
        A_hat = torch.eye(n_nodes, dtype=torch.float32)
        A_hat[edge_index[0], edge_index[1]] = 1
        # TODO which dimension to sum?
        D = torch.sum(A_hat, dim=0)
        D_hat_ = D ** (-1/2)
        D_hat = torch.diag(D_hat_)

        #D_hat__ = torch.nan_to_num(D_hat_, posinf=0.0, neginf=0.0)

        #mmr = torch.mm(D_hat, A_hat)
        #mmr = torch.mm(mmr, D_hat)
        #mmr = torch.mm(mmr, x)
        #x_ = torch.mm(mmr, self.theta)

        sym_norm = torch.mm(torch.mm(D_hat, A_hat), D_hat)
        x_theta = torch.mm(x, self.theta)

        x_ = torch.mm(sym_norm, x_theta)

        return x_


from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        """
        TODO:
            1. Define the first convolution layer using `GCNConv()`. Set `out_channels` to 64;
            2. Define the first activation layer using `nn.ReLU()`;
            3. Define the second convolution layer using `GCNConv()`. Set `out_channels` to 64;
            4. Define the second activation layer using `nn.ReLU()`;
            5. Define the third convolution layer using `GCNConv()`. Set `out_channels` to 64;
            6. Define the dropout layer using `nn.Dropout()`;
            7. Define the linear layer using `nn.Linear()`. Set `output_size` to 2.

        Note that for MUTAG dataset, the number of node features is 7, and the number of classes is 2.

        """

        # your code here
        self.gcn_conv1 = GCNConv(7, 64)
        self.relu1 = nn.ReLU()
        self.gcn_conv2 = GCNConv(64, 64)
        self.relu2 = nn.ReLU()
        self.gcn_conv3 = GCNConv(64, 64)
        self.drop = nn.Dropout()
        self.linear = nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
        """
        TODO:
            1. Pass the data through the frst convolution layer;
            2. Pass the data through the activation layer;
            3. Pass the data through the second convolution layer;
            4. Obtain the graph embeddings using the readout layer with `global_mean_pool()`;
            5. Pass the graph embeddgins through the dropout layer;
            6. Pass the graph embeddings through the linear layer.

        Arguments:
            x: [num_nodes, 7], node features
            edge_index: [2, num_edges], edges
            batch: [num_nodes], batch assignment vector which maps each node to its
                   respective graph in the batch

        Outputs:
            probs: probabilities of shape (batch_size, 2)
        """

        # your code here
        out = self.gcn_conv1(x, edge_index)
        out = self.relu1(out)
        out = self.gcn_conv2(out, edge_index)
        out = self.relu2(out)
        out = self.gcn_conv3(out, edge_index)

        g_embeddings = global_mean_pool(out, batch=batch)

        g_embeddings = self.drop(g_embeddings)

        probs = self.linear(g_embeddings)
        # TODO do I need to apply sigmoid ? NO, because we used CELoss no BCELoss
        return probs


gcn = GCN()

# optimizer
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.02)
# loss
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader, epoch):
    gcn.train()
    curr_epoch_loss = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        """
        TODO: train the model for one epoch.
        
        Note that you can acess the batch data using `data.x`, `data.edge_index`, `data.batch`, `data,y`.
        """

        # your code here
        optimizer.zero_grad()
        y_hat = gcn(data.x, data.edge_index, data.batch)
        loss = criterion(y_hat, data.y)
        loss.backward()
        optimizer.step()

        curr_epoch_loss.append(loss.cpu().data.numpy())
    print(f"epoch{epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")

def test(loader):
    gcn.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = gcn(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(200):
    train(train_loader, epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')