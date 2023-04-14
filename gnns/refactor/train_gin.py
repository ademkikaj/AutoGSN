import argparse
import json
import logging
import os
import random
from shutil import rmtree
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from models.gcn_alternatives import GINDefault, GINPaper
from utilities.preprocess_features import preprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True, default="1")
parser.add_argument("-d", "--dataset", required=True, default="MUTAG")
parser.add_argument("-b", "--batch", required=True, default=32)
parser.add_argument("-hl", "--hidden", required=True, default=16)
args = parser.parse_args()


logging.basicConfig(
    filename="results_" + args.dataset + "_" + args.seed + "_" + args.batch + "_" + args.hidden + ".log",
    filemode="w",
    format="%(message)s",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


seed = int(args.seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# rmtree(os.path.join("data", args.dataset, "processed"))
print("Processing started...")
start_time = time.time()
dataset = TUDataset(root="data", name=args.dataset, pre_transform=preprocess).shuffle()
print("--- Total %s seconds ---" % (time.time() - start_time))
print("Processing finished...")
# exit()
# dataset = TUDataset(root="data", name=args.dataset).shuffle()


train_dataset = dataset[: int(len(dataset) * 0.9)]
test_dataset = dataset[int(len(dataset) * 0.9) :]
print("Training set: %d" % len(train_dataset))
print("Test set: %d" % len(test_dataset))


batch_size = int(args.batch)
hidden_channels = int(args.hidden)
criterion = torch.nn.CrossEntropyLoss()
model = GINPaper(
    in_channels=dataset.num_node_features,
    hidden_channels=hidden_channels,
    num_layers=4,
    out_channels=dataset.num_classes,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(model, train_loader, optimizer):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data.to(device)

        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.

        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model, test_loader):
    model.eval()

    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.to(device)

        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 351):
    print("EPOCH %d/%d" % (epoch, 350))
    train(model, train_data, optimizer)
    train_acc = test(model, train_data)
    test_acc = test(model, test_data)
    print(
        f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}",
    )
    x = {
        "Epoch": float("{:.2f}".format(float(epoch))),
        "Train Acc": float("{:>5.2f}".format(float(train_acc * 100))),
        "Test Acc": float("{:.2f}".format(float(test_acc * 100))),
    }
    logger.info(json.dumps(x))
