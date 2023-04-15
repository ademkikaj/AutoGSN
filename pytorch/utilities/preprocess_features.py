from collections import Counter
import pickle, networkx as nx
import torch
import torch_geometric
import matplotlib.pyplot as plt

features = pickle.load(open("data_features/NCI1_3699.dat", "rb"))
nx_graphs = pickle.load(open("data_networkx/NCI1_labeled_nr.dat", "rb"))
graph_index = 0


def plot(graph: nx.Graph, feature):
    print(graph.edges)

    plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    edge_labels = {}
    for edge in graph.edges:
        edge_labels[(edge[0], edge[1])] = graph.get_edge_data(edge[0], edge[1])["label"]

    nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    plt.figure(2)
    pos = nx.spring_layout(feature)
    nx.draw(feature, pos)
    nx.draw_networkx_nodes(feature, pos)
    nx.draw_networkx_edges(feature, pos)
    nx.draw_networkx_labels(feature, pos)
    nx.draw_networkx_edge_labels(feature, pos)


def preprocess(graph):
    global graph_index
    nx_graph: nx.Graph = nx_graphs[graph_index]
    for feature in features:
        # plot(nx_graph, feature)
        counter = Counter()
        subgraphs = nx.isomorphism.GraphMatcher(
            nx_graph,
            feature,
            node_match=nx.isomorphism.categorical_node_match("label", None),
            edge_match=nx.isomorphism.categorical_edge_match("label", None),
        ).subgraph_isomorphisms_iter()
        # subgraphs = nx.isomorphism.GraphMatcher(
        #     nx_graph,
        #     feature
        # ).subgraph_isomorphisms_iter()
        subgraphs = [tuple((subgraph.keys())) for subgraph in subgraphs]
        unpacked = []
        for k in subgraphs:
            unpacked.append(list(k))
        counter = Counter()
        unpacked = [item for sublist in unpacked for item in sublist]
        counter.update(unpacked)
        # print(counter)
        feature_tensor = torch.zeros(nx_graph.number_of_nodes(), 1)
        for k, v in counter.items():
            feature_tensor[k] = v
        graph.x = torch.cat((graph.x, feature_tensor), 1)
        # print(graph.x)
    graph_index += 1
    return graph
