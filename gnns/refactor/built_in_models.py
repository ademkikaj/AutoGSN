import torch, os, random, json, logging, argparse
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GIN, Linear, global_add_pool
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True)
parser.add_argument("-d", "--dataset", required=True)
args = parser.parse_args()

logging.basicConfig(
    filename="results_" + args.dataset + "_" + args.seed + ".log",
    filemode="w",
    format="%(message)s",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

seed = int(args.seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset_name = args.dataset
data_root = os.path.join(os.path.dirname(__file__), "data", dataset_name)
dataset = TUDataset(root=data_root, name=dataset_name).shuffle()

k_cross = 10
batch_size = 32
learning_rate = 0.01
hidden_channels = 32
weight_decay = 0.0005
nr_epochs = 2
parameters_dict = {
    "k_cross": k_cross,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "hidden_channels": hidden_channels,
    "weight_decay": weight_decay,
    "nr_epochs": nr_epochs,
    "seed": seed,
    "dataset": dataset_name,
}

in_channels = dataset.num_node_features
out_channels = dataset.num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GIN(in_channels, hidden_channels, 5).to(device)


train_test_dataset = dataset[: int(len(dataset) * 0.9)]
val_dataset = dataset[int(len(dataset) * 0.9) :]
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
criterion = torch.nn.CrossEntropyLoss()
linear = Linear(hidden_channels, out_channels)


def get_folds(nr_of_folds):
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

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        folds.append((train_loader, test_loader))

    return folds


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def test(test_data):
    model.eval()
    loss = 0
    acc = 0

    for batch in test_data:
        h = model(batch.x, batch.edge_index)

        hG = global_add_pool(h, batch.batch)

        h = F.dropout(hG, p=0.5, training=model.training)
        h = linear(h)
        out = F.log_softmax(h, dim=1)

        loss += criterion(out, batch.y) / len(test_data)
        acc += accuracy(out.argmax(dim=1), batch.y) / len(test_data)

    return loss, acc


def train():
    model.train()
    folds = get_folds(k_cross)
    epochs_res = []
    val_res = []
    for epoch in range(nr_epochs):
        for fold in folds:
            total_loss = 0
            acc = 0
            train_data = fold[0]
            test_data = fold[1]
            for batch in train_data:
                optimizer.zero_grad()
                h = model(batch.x, batch.edge_index)

                hG = global_add_pool(h, batch.batch)

                h = F.dropout(hG, p=0.5, training=model.training)
                h = linear(h)
                out = F.log_softmax(h, dim=1)

                loss = criterion(out, batch.y)
                total_loss += loss / len(train_data)

                acc += accuracy(out.argmax(dim=1), batch.y) / len(train_data)
                loss.backward()
                optimizer.step()

                test_loss, test_acc = test(test_data)
            epochs_res.append(
                {
                    "Epoch": epoch,
                    "Train Loss": float("{:.2f}".format(float(total_loss))),
                    "Train Acc": float("{:>5.2f}".format(float(acc * 100))),
                    "Test Loss": float("{:.2f}".format(float(test_loss))),
                    "Test Acc": float("{:.2f}".format(float(test_acc * 100))),
                }
            )

        test_loss, test_acc = test(val_loader)
        val_res.append(
            {
                "Val Loss: ": float("{:.2f}".format(float(test_loss))),
                "Val Acc: ": float("{:.2f}".format(float(test_acc * 100))),
            }
        )
    logger.info(
        json.dumps(
            {
                "PARAMETERS": parameters_dict,
                "EPOCHS": epochs_res,
                "VALIDATIONS": val_res,
            }
        )
    )


train()
