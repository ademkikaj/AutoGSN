from operator import itemgetter
import torch
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

node_counter = 0
all_graphs = []


for data in dataset:
    if graph_node_feature and graph_edge_feature:
        g: Graph = torch_geometric.utils.to_networkx(
            data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True
        )
        for node in g.nodes:
            g.add_node(
                node, label=max(enumerate(g.nodes[node]["x"]), key=itemgetter(1))[0]
            )
        for edge in g.edges:
            u = edge[0]
            v = edge[1]
            g.add_edge(
                u,
                v,
                label=max(
                    enumerate(g.get_edge_data(u, v)["edge_attr"]), key=itemgetter(1)
                )[0],
            )
        all_graphs.append(g)
    elif graph_node_feature:
        g: Graph = torch_geometric.utils.to_networkx(
            data, node_attrs=["x"], to_undirected=True
        )
        for node in g.nodes:
            g.add_node(
                node, label=max(enumerate(g.nodes[node]["x"]), key=itemgetter(1))[0]
            )
        for i, edge in enumerate(g.edges):
            g.add_edge(edge[0], edge[1], label=0)
        all_graphs.append(g)
    elif graph_edge_feature:
        g: Graph = torch_geometric.utils.to_networkx(
            data, edge_attrs=["edge_attr"], to_undirected=True
        )
        for node in g.nodes:
            g.add_node(node, label=0)
        for edge in g.edges:
            u = edge[0]
            v = edge[1]
            g.add_edge(
                u,
                v,
                label=max(
                    enumerate(g.get_edge_data(u, v)["edge_attr"]), key=itemgetter(1)
                )[0],
            )
    else:
        g: Graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        for node in g.nodes:
            g.add_node(node, label=0)
        for i, edge in enumerate(g.edges):
            g.add_edge(edge[0], edge[1], label=0)
        all_graphs.append(g)
    node_counter += len(data.x)

with open(data_save, "wb") as file:
    pickle.dump(all_graphs, file)
