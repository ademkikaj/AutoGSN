import sklearn.feature_selection as fs
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import networkx as nx


dataset = "MUTAG_labeled_nr"
dataset_labels = "MUTAG_y"
features = "MUTAG_4"


def save_features(file_name, features_df: pd.DataFrame):
    features_df.to_pickle(
        os.path.join(
            os.path.dirname(__file__),
            file_name + "_df.dat",
        )
    )


def load_labels(dataset):
    graph_labels = os.path.join(
        os.path.dirname(__file__),
        dataset + ".dat",
    )

    return pickle.load(open(graph_labels, "rb"))


def load_dataset(dataset):
    graphs = os.path.join(
        os.path.dirname(__file__),
        dataset + ".dat",
    )

    return pickle.load(open(graphs, "rb"))


def load_graph_features(features):
    graphs = os.path.join(
        os.path.dirname(__file__),
        features + ".dat",
    )

    return pickle.load(open(graphs, "rb"))


def plot_graph(graph_pos: nx.Graph, graph_neg: nx.Graph):
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(graph_pos, seed=0)
    nx.draw_networkx_nodes(graph_pos, pos, graph_pos.nodes)
    nx.draw_networkx_edges(graph_pos, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(
        graph_pos, pos, labels=nx.get_node_attributes(graph_pos, "label")
    )
    nx.draw_networkx_edge_labels(
        graph_pos, pos, edge_labels=nx.get_edge_attributes(graph_pos, "label")
    )

    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(graph_neg, seed=0)
    nx.draw_networkx_nodes(graph_neg, pos, graph_neg.nodes)
    nx.draw_networkx_edges(graph_neg, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(
        graph_neg, pos, labels=nx.get_node_attributes(graph_neg, "label")
    )
    nx.draw_networkx_edge_labels(
        graph_neg, pos, edge_labels=nx.get_edge_attributes(graph_neg, "label")
    )

    plt.show()


def subgraph_exists(graph, subgraph):
    return nx.isomorphism.GraphMatcher(
        graph,
        subgraph,
        node_match=nx.isomorphism.categorical_node_match("label", None),
        edge_match=nx.isomorphism.categorical_edge_match("label", None),
    ).subgraph_is_isomorphic()


graph_features = load_graph_features(features)
print("Loaded %d features" % len(graph_features))
dataset = load_dataset(dataset)
dataset_labels = load_labels(dataset_labels)
df = {}
for k, graph in enumerate(dataset):
    print("%d/%d" % (k, len(dataset)))
    for i, subgraph in enumerate(graph_features):
        try:
            df[i].append(int(subgraph_exists(graph, subgraph)))
        except:
            df[i] = [int(subgraph_exists(graph, subgraph))]

df["y"] = dataset_labels

df = pd.DataFrame(df)
save_features(features, df)

print(df.head())
x = df.drop("y", axis=1)
y = df["y"]
chi_scores = fs.chi2(x, y)

p_values = pd.Series(chi_scores[1])
p_values.sort_values(ascending=False, inplace=True)
p_values.plot.bar()
plt.show()
