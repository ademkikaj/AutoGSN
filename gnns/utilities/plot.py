import networkx as nx
import matplotlib.pyplot as plt


def plot(graph: nx.Graph):
    plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos)
    plt.show()
