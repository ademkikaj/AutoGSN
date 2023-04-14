import torch, os, random, json, logging, argparse, pickle
import torch_geometric
from torch_geometric.datasets import TUDataset
import networkx as nx
from utilities.graph_functions import count_cliques, get_cycles_of_size
import matplotlib.pyplot as plt

dataset_name = "PROTEINS"
data_root = os.path.join(os.path.dirname(__file__),'data',dataset_name)
dataset = TUDataset(root=data_root, name=dataset_name)

all_graphs = []



for data in dataset:
    g: nx.Graph  = torch_geometric.utils.to_networkx(data, to_undirected=True)
    all_graphs.append(g)
    
nr_of_graphs = len(all_graphs)
print(nr_of_graphs)

res = count_cliques(nr_of_graphs, all_graphs, 3, 20)
print(len(res))
file_name = dataset_name+"_cliques_processed.dat"
with open(file_name, "wb") as file:
        pickle.dump(res, file)

res = get_cycles_of_size(nr_of_graphs, all_graphs)
print(len(res))

file_name = dataset_name+"_cycles_processed.dat"
with open(file_name, "wb") as file:
        pickle.dump(res, file)
