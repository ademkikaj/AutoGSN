import matplotlib.pyplot as plt
import pickle
import networkx as nx


def single_plot(graph):
    plot = graph
    pos = nx.spring_layout(plot, seed=0)
    nx.draw_networkx_nodes(plot, pos, plot.nodes)
    nx.draw_networkx_edges(plot, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(plot, pos, labels=nx.get_node_attributes(plot, "label"))
    nx.draw_networkx_edge_labels(
        plot, pos, edge_labels=nx.get_edge_attributes(plot, "label")
    )
    plt.show()


def loop_plot(plots, title):
    figs = {}
    axs = {}
    count = 1
    f = plt.figure(figsize=(10, 10))
    plt.title(title)
    for idx, plot in enumerate(plots):
        f.add_subplot(2, int(len(plots) / 2), idx + 1)
        pos = nx.spring_layout(plot, seed=0)
        nx.draw_networkx_nodes(plot, pos, plot.nodes)
        nx.draw_networkx_edges(plot, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(plot, pos, labels=nx.get_node_attributes(plot, "label"))
        nx.draw_networkx_edge_labels(
            plot, pos, edge_labels=nx.get_edge_attributes(plot, "label")
        )
        if count % 3 == 0:
            count = 0
        else:
            count += 1


features = pickle.load(open("mutag_playground/gspan_90.dat", "rb"))

for i, f in enumerate(features):
    single_plot(f)
exit()
features = pickle.load(
    open("mutag_playground/MUTAG_50_xgb_6_selected_features.dat", "rb")
)
loop_plot(features, "XGB")

plt.show()
