import torch_geometric, os, pickle, networkx as nx
from torch_geometric.datasets import TUDataset
from networkx import Graph
from plot import plot

dataset = "NCI1"
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset)
data_save = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data_networkx", dataset + ".dat"
)

dataset = TUDataset(root=data_root, name=dataset)

all_graphs = []

for data in dataset:
    try:
        g: Graph = torch_geometric.utils.to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=True,
        )
    except:
        g: Graph = torch_geometric.utils.to_networkx(
            data,
            node_attrs=["x"],
            to_undirected=True,
        )
    # print(nx.get_node_attributes(g, "x"))
    # print(g.nodes(data=True)[0])
    # print(nx.get_edge_attributes(g, "edge_attr"))
    all_graphs.append(g)
print(len(all_graphs))
with open(data_save, "wb") as file:
    pickle.dump(all_graphs, file)
