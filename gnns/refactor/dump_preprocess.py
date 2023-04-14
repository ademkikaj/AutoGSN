import os, torch
from shutil import rmtree
import pickle
import numpy as np
import random
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GIN

model = GIN(2, 6, 5)


seed = 0

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# data_name = 'Mutagenicity'
# data_name = 'REDDIT-BINARY'
data_name = 'IMDB-BINARY'
data_name = 'PROTEINS'
file_name = data_name + "_cliques_processed.dat"
cycles = data_name + "_cycles_processed.dat"
# [3 4 5 6 7 8 9]
# [0 1 2 3 4 5 6]
# res_file_name = "res_" + data_name + "3_9_seed_1_500_epochs" + ".dat"

proccesed_load_cliques = pickle.load(open(file_name, "rb"))
proccesed_load_cycles = pickle.load(open(cycles, "rb"))

def get_features(graph, curr_features):
    feature_cliques = proccesed_load_cliques[graph]
    tensor_all_features_cliques = torch.empty(0, len(list(feature_cliques[0])))
    for _, features in feature_cliques.items():
        tensor_all_features_cliques = torch.cat((tensor_all_features_cliques, torch.FloatTensor([list(features.values())])))
    
    feature_cycles = proccesed_load_cycles[graph]
    
    if len(feature_cliques) != len(feature_cycles):
        print(graph)
    
    tensor_all_features_cycles = torch.empty(0, len(list(feature_cycles[0])))
    for _, features in feature_cycles.items():
        tensor_all_features_cycles = torch.cat((tensor_all_features_cycles, torch.FloatTensor([list(features.values())])))
    
    return torch.cat((curr_features, tensor_all_features_cliques, tensor_all_features_cycles), 1)



data_root = os.path.join(os.path.dirname(__file__),'data',data_name)
# print(data_root)
rmtree(os.path.join(data_root,data_name,"processed"))
i=0
def transform_node_feature(graph):
    global i
    # graph.x = torch.tensor([[1]]*graph.num_nodes, dtype=torch.float)
    graph.x = get_features(i, graph.x)
    i+=1
    return graph

dataset = TUDataset(root=data_root, name=data_name, pre_transform=transform_node_feature).shuffle()

train_dataset = dataset[:int(len(dataset)*0.8)]
# val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.8):]

print(f'Training set   = {len(train_dataset)} graphs')
# print(f'Validation set = {len(val_dataset)} graphs')
print(f'Test set       = {len(test_dataset)} graphs')

# Create mini-batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    
def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.001,
                                      weight_decay=0.05)
    epochs = 350

    model.train()
    print("Training started...")
    val_score = []
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

            # Validation
            # val_loss, val_acc = test(model, val_loader)
            test_loss, test_acc = test(model, test_loader)
    
    # if(epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
              f'| Train Acc: {acc*100:>5.2f}% '
              f'| Val Loss: {test_loss:.2f} '
              f'| Val Acc: {test_acc*100:.2f}%')
        # val_score.append(val_acc*100)
        
    # with open((res_file_name), "wb") as file_pickle:
    #     pickle.dump(val_score, file_pickle)
          
    
    
    return model

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


# gin = GIN(dim_h=64)
# gin = train(gin,train_loader)

# gcn = GCN(dim_h=64)
# gcn = train(gcn, train_loader)