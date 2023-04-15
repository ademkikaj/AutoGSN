import torch_geometric, os, pickle, networkx as nx
from torch_geometric.datasets import TUDataset
from networkx import Graph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
args = parser.parse_args()

dataset = args.dataset
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset)
data_save = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data_networkx", dataset + ".dat"
)

dataset = TUDataset(root=data_root, name=dataset)

tmp_graph = dataset[0]
graph_node_feature = False
graph_edge_feature = False

if "x" in tmp_graph:
    graph_node_feature = True
if "edge_attr" in tmp_graph:
    graph_edge_feature = True

all_graphs = []

for data in dataset:
    g: Graph = None
    if graph_node_feature and graph_edge_feature:
        g: Graph = torch_geometric.utils.to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=True,
        )
    elif graph_node_feature:
        g: Graph = torch_geometric.utils.to_networkx(
            data,
            node_attrs=["x"],
            to_undirected=True,
        )
    elif graph_edge_feature:
        g: Graph = torch_geometric.utils.to_networkx(
            data,
            edge_attrs=["edge_attr"],
            to_undirected=True,
        )
    all_graphs.append(g)
print(len(all_graphs))
with open(data_save, "wb") as file:
    pickle.dump(all_graphs, file)
