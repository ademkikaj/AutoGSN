import torch
from torch_geometric.datasets import TUDataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, GCN, GIN
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import Subset
import pickle

dataset = TUDataset(root="data/TUDataset", name="PROTEINS")
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_test_dataset = dataset[: int(len(dataset) * 0.8)]
validation_dataset = dataset[int(len(dataset) * 0.8) :]
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)


# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = Linear(hidden_channels, dataset.num_classes)

#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)

#         # 2. Readout layer
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

#         # 3. Apply a final classifier
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)

#         return x


model = GCN(in_channels=dataset.num_node_features, hidden_channels=64, num_layers=3)
linear = Linear(64, dataset.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def get_folds(dataset, nr_of_folds, batch_size):
    total_size = len(dataset)
    fraction = 1 / nr_of_folds
    seg = int(total_size * fraction)
    folds = []
    for i in range(nr_of_folds):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        folds.append((train_loader, test_loader))

    return folds


def train(train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        x = model(data.x, data.edge_index)  # Perform a single forward pass.
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=model.training)
        out = linear(x)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(test_loader):
    model.eval()

    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        x = model(data.x, data.edge_index)  # Perform a single forward pass.
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=model.training)
        out = linear(x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.


folds = get_folds(train_test_dataset, 10, 64)

for epoch in range(1, 351):
    print("EPOCH %d/%d" % (epoch, 350))
    train_acc_agg = []
    for fold in folds:
        train_loader = fold[0]
        test_loader = fold[1]
        train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        val_acc = test(validation_loader)
        # print(
        #     f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}",
        # )
        train_acc_agg.append(train_acc)

    with open("train_acc.dat", "wb") as file:
        pickle.dump(train_acc, file)
