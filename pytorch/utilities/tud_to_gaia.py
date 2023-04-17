import torch_geometric, os, pickle
import argparse
from operator import itemgetter
from torch_geometric.datasets import TUDataset
from networkx import Graph

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
args = parser.parse_args()

dataset = args.dataset
node_file = dataset + "_node_file.txt"
edge_file = dataset + "_edge_file.txt"
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset)
data_root_node_out = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data_gaia", node_file
)
data_root_edge_out = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data_gaia", edge_file
)
node_file_w = open(data_root_node_out, "w")
edge_file_w = open(data_root_edge_out, "w")

dataset = TUDataset(root=data_root, name=dataset)

tmp_graph = dataset[0]
graph_node_feature = False
graph_edge_feature = False
nr_node_labels = 0
nr_edge_labels = 0

if "x" in tmp_graph:
    graph_node_feature = True
    nr_node_labels = tmp_graph.x.shape[1]
if "edge_attr" in tmp_graph:
    graph_edge_feature = True
    nr_edge_labels = tmp_graph.edge_attr.shape[1]

pos_graphs = []
neg_graphs = []

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
        if data.y == 1:
            pos_graphs.append(g)
        if data.y == 0:
            neg_graphs.append(g)
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
        if data.y == 1:
            pos_graphs.append(g)
        if data.y == 0:
            neg_graphs.append(g)
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
        if data.y == 1:
            pos_graphs.append(g)
        if data.y == 0:
            neg_graphs.append(g)

print("Number of positive graphs: %d" % len(pos_graphs))

graph_counter = 0
for graph in pos_graphs:
    graph: Graph
    for node in graph.nodes:
        curr_node = "%dg %d %d %d\n" % (
            graph_counter,
            graph_counter,
            node,
            int(graph.nodes[node]["label"]) + 1,
        )
        node_file_w.write(curr_node)
    for edge in graph.edges:
        u = edge[0]
        v = edge[1]
        curr_edge = "%dg %d %d %d %d\n" % (
            graph_counter,
            graph_counter,
            u,
            v,
            int(graph.get_edge_data(u, v)["label"]) + nr_node_labels + 1,
        )
        edge_file_w.write(curr_edge)
    graph_counter += 1


for graph in neg_graphs:
    graph: Graph
    for node in graph.nodes:
        curr_node = "%dg %d %d %d\n" % (
            graph_counter,
            graph_counter,
            node,
            int(graph.nodes[node]["label"]) + 1,
        )
        node_file_w.write(curr_node)
    for edge in graph.edges:
        u = edge[0]
        v = edge[1]
        curr_edge = "%dg %d %d %d %d\n" % (
            graph_counter,
            graph_counter,
            u,
            v,
            int(graph.get_edge_data(u, v)["label"]) + nr_node_labels + 1,
        )
        edge_file_w.write(curr_edge)
    graph_counter += 1

node_file_w.close()
edge_file_w.close()
