import networkx as nx, matplotlib.pyplot as plt, pickle, os


def save(type, graphs):
    data_write = os.path.join(
        os.path.dirname(__file__),
        "data_static_features",
        type + ".dat",
    )
    with open(data_write, "wb") as write_file:
        pickle.dump(graphs, write_file)


def plot(graph: nx.Graph):
    plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.show()


def cycle(k) -> nx.Graph:
    return nx.cycle_graph(k)


def clique(k) -> nx.Graph:
    return nx.complete_graph(k)


def path(k) -> nx.Graph:
    return nx.path_graph(k)


graph = clique(4)
save("clique4", [graph])
