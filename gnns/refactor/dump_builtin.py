import os
import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN, GraphSAGE, GIN, Linear, global_mean_pool, global_max_pool, global_add_pool, summary
import torch.nn.functional as F


seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

data_name = "IMDB-BINARY"
data_name = "REDDIT-BINARY"
data_name = "PROTEINS"
data_root = os.path.join(os.path.dirname(__file__), "data", data_name)

def transform_node_feature(graph):
    # graph.x = torch.tensor([[1]]*graph.num_nodes, dtype=torch.float)
    return graph

dataset = TUDataset(root=data_root, name=data_name, pre_transform=transform_node_feature).shuffle()

train_dataset = dataset[:int(len(dataset)*0.8)]
# val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.8):]

print(f'Training set   = {len(train_dataset)} graphs')
# print(f'Validation set = {len(val_dataset)} graphs')
print(f'Test set       = {len(test_dataset)} graphs')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = GCN(dataset.num_node_features, 64, 5)
model = GraphSAGE(dataset.num_node_features, 64, 5)
model = GIN(dataset.num_node_features, 64, 4, dataset.num_classes)
# print(summary(model, dataset[0].x, dataset[0].edge_index))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    for epoch in range(1, 350):

        total_loss = 0
        acc = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)  # Perform a single forward pass.
            # Graph-level readout
            hG = global_add_pool(out, data.batch)
            # # # Classifier
            h = F.dropout(hG, p=0.5, training=True)
            # # lin = Linear(64, dataset.num_classes)
            # h = lin(h)
            out = F.log_softmax(h, dim=1)
            loss = criterion(out, data.y)  # Compute the loss.
            total_loss += loss / len(train_loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(train_loader)
            loss.backward()
            optimizer.step()

            # Validation
            # val_loss, val_acc = test(model, val_loader)
            test_loss, test_acc = test(model, test_loader)
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
            f'| Train Acc: {acc*100:>5.2f}% '
            f'| Val Loss: {test_loss:.2f} '
            f'| Val Acc: {test_acc*100:.2f}%')
        # loss.backward()  # Derive gradients.
        # optimizer.step()  # Update parameters based on gradients.
        # optimizer.zero_grad()  # Clear gradients.


# def test(loader):
#     model.eval()

#     correct = 0
#     for data in loader:  # Iterate in batches over the training/test dataset.
#         out = model(data.x, data.edge_index)
#         # Graph-level readout
#         hG = global_add_pool(out, data.batch)
#         # # Classifier
#         h = F.dropout(hG, p=0.5, training=True)
#         lin = Linear(64, dataset.num_classes)
#         h = lin(h)
#         out = F.log_softmax(h, dim=1)
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        # Graph-level readout
        hG = global_add_pool(out, data.batch)
        # # Classifier
        h = F.dropout(hG, p=0.5, training=True)
        # lin = Linear(64, dataset.num_classes)
        # h = lin(h)
        out = F.log_softmax(h, dim=1)
        
        loss += criterion(out, data.y) / len(loader) # Compute the loss.
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

train()