import torch, os, random, json, logging, argparse
from models.gcn_alternatives import GCN
from models.gin import GIN
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True)
parser.add_argument("-d", "--dataset", required=True)
args = parser.parse_args()

logging.basicConfig(filename='results_'+args.dataset+'_'+args.seed+'.log', filemode='w', format='%(message)s')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model: torch.nn.Module, train_data, test_data, nr_epochs, optimizer: torch.optim.Optimizer, criterion, device):
    model.train()
    print("Training started...")
    test_score = []
    for epoch in range(nr_epochs+1):
        total_loss = 0
        acc = 0
        for data in train_data:
            data.to(device)
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(train_data)
            acc += accuracy(out.argmax(dim=1), data.y) / len(train_data)
            loss.backward()
            optimizer.step()
            test_loss, test_acc = test(model, test_data, criterion)
        x = {'Epoch': epoch, 'Train Loss': float("{:.2f}".format(float(total_loss))), 
             'Train Acc' : float("{:>5.2f}".format(float(acc*100))), 
             'Test Loss': float("{:.2f}".format(float(test_loss))), 
             'Test Acc': float("{:.2f}".format(float(test_acc*100)))}
        y = json.dumps(x)
        logger.info(y)
        test_score.append(test_acc)
    return max(test_score)

@torch.no_grad()
def test(model, test_data, criterion):
    model.eval()
    loss = 0
    acc = 0

    for data in test_data:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(test_data)
        acc += accuracy(out.argmax(dim=1), data.y) / len(test_data)

    return loss, acc

def crossvalid(dataset=None, k_fold=10):    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    sets = []
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" % (trll,trlr,trrl,trrr,vall,valr))
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        test_indices = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        test_set = torch.utils.data.dataset.Subset(dataset, test_indices)  
        
        sets.append((train_set, test_set))
    return sets

seed = int(args.seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = args.dataset
data_root = os.path.join(os.path.dirname(__file__),'data',dataset)
dataset = TUDataset(root=data_root, name=dataset).shuffle()

k_cross = 10
batch_size = 32
hidden_channels = None
learning_rate = 0.01
weight_decay = 0.5
nr_epochs = 350

in_channels = dataset.num_node_features
out_channels = dataset.num_classes
hidden_channels = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GCN(in_channels, hidden_channels, out_channels, aggr="add").to(device)
# model = GIN(in_channels, hidden_channels, out_channels).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset  = dataset[int(len(dataset)*0.8):]
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
max_acc = train(model, train_data, test_data, nr_epochs, optimizer, criterion, device)
print(max_acc)

# folds = crossvalid(dataset)
# for fold in folds:
    # train_data = DataLoader(fold[0], batch_size=batch_size, shuffle=True)
    # test_data = DataLoader(fold[1], batch_size=batch_size, shuffle=True)
    # max_acc = train(model, train_data, test_data, nr_epochs, optimizer, criterion)
    # print(max_acc)



