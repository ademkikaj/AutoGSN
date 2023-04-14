import networkx as nx, os, pickle, matplotlib.pyplot as plt, random

random.seed(1)


def plot_graph(graph1: nx.Graph):
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(graph1)
    nx.draw_networkx_nodes(graph1, pos, graph1.nodes)
    nx.draw_networkx_edges(graph1, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph1, pos, labels=nx.get_node_attributes(graph1, "label"))
    nx.draw_networkx_edge_labels(
        graph1, pos, edge_labels=nx.get_edge_attributes(graph1, "label")
    )
    plt.show()


def plot_graph_2(graph1: nx.Graph, graph2: nx.Graph):
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(graph1)
    nx.draw_networkx_nodes(graph1, pos, graph1.nodes)
    nx.draw_networkx_edges(graph1, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph1, pos, labels=nx.get_node_attributes(graph1, "label"))
    nx.draw_networkx_edge_labels(
        graph1, pos, edge_labels=nx.get_edge_attributes(graph1, "label")
    )

    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(graph2)
    nx.draw_networkx_nodes(graph2, pos, graph2.nodes)
    nx.draw_networkx_edges(graph2, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph2, pos, labels=nx.get_node_attributes(graph2, "label"))
    nx.draw_networkx_edge_labels(
        graph2, pos, edge_labels=nx.get_edge_attributes(graph2, "label")
    )

    plt.show()


dataset = "PROTEINS"

data = os.path.join(os.path.dirname(__file__), "data_nx", dataset + "_labeled_nr.dat")

data = os.path.join(os.path.dirname(__file__), "data_nx_features", dataset + "_pos.dat")
with open(data, "rb") as data_file:
    graphs_pos = pickle.load(data_file)

data = os.path.join(os.path.dirname(__file__), "data_nx_features", dataset + "_neg.dat")
with open(data, "rb") as data_file:
    graphs_neg = pickle.load(data_file)

print(graphs_pos[0].nodes(data=True))
exit()

for i, g in enumerate(graphs_pos):
    print(
        nx.weisfeiler_lehman_graph_hash(graphs_pos[0])
        == nx.weisfeiler_lehman_graph_hash(graphs_neg[0])
    )
