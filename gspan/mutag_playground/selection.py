import argparse
import sklearn.feature_selection as fs
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import networkx as nx

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def pickle_features(dataset, features):
    selected_features = os.path.join(os.path.dirname(__file__), dataset + "_.dat")
    pickle.dump(features, open(selected_features, "wb"))


def load_df(dataset):
    return pickle.load(open(dataset, "rb"))


def chi_square_features(train, labels, top_k, features_graph):
    chi_scores = fs.chi2(train, labels)
    p_values = pd.Series(chi_scores[1], index=train.columns)
    p_values.sort_values(ascending=True, inplace=True)
    p_values = p_values[-top_k:]
    selected_graphs = []
    for key in p_values.keys():
        # print(key)
        selected_graphs.append(features_graph[key])
    return selected_graphs


def random_forest_features(train, label, top_k, features_graph):
    forest = RandomForestClassifier(random_state=0)
    forest.fit(train, label)
    feature_names = [f"{i}" for i in range(train.shape[1])]
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances.sort_values(ascending=True, inplace=True)
    forest_importances = forest_importances[-top_k:]
    selected_graphs = []
    for key in list(forest_importances.keys()):
        selected_graphs.append(features_graph[int(key)])
    return selected_graphs


def xgb_features(train, label, top_k, features_graph):
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(
        train, label, test_size=test_size, random_state=7
    )
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # plot_importance(model)
    # plt.show()
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    feature_important = model.get_booster().get_score(importance_type="weight")
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(
        by="score", ascending=False
    )
    # print(data.head(top_k))
    selected_graphs = []
    for key in list(data.index[:top_k]):
        selected_graphs.append(features_graph[int(key)])
    return selected_graphs


def mutual_info(train, label, top_k, features_graph):
    mutual = fs.mutual_info_classif(train, label)
    mutual = pd.Series(mutual)
    mutual.sort_values(ascending=False, inplace=True)
    selected_graphs = []
    for key in mutual[:top_k].keys():
        # print(key)
        selected_graphs.append(features_graph[key])
    return selected_graphs


def save_features(flatten_graph, features_df: pd.DataFrame):
    features_df.to_pickle(flatten_graph)


def load_labels(dataset):
    graph_labels = os.path.join(
        os.path.dirname(__file__),
        dataset + ".dat",
    )

    return pickle.load(open(graph_labels, "rb"))


def load_all_graphs(dataset):
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


def perform_selection(
    dataframe_name, df, feature_selection_input, top_k_features, graph_features
):
    train = df.drop("y", axis=1)
    label = df["y"]
    if feature_selection_input == "xgb":
        res = xgb_features(train, label, top_k_features, graph_features)
    elif feature_selection_input == "rf":
        res = random_forest_features(train, label, top_k_features, graph_features)
    elif feature_selection_input == "chi2":
        res = chi_square_features(train, label, top_k_features, graph_features)
    elif feature_selection_input == "mi":
        res = mutual_info(train, label, top_k_features, graph_features)

    pickle_features(
        dataframe_name + "_" + feature_selection_input + "_" + top_k_features, res
    )


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-f", "--features", required=True)
parser.add_argument(
    "-s", "--selection", required=True, choices=["xgb", "rf", "chi2", "mi"]
)
parser.add_argument("-k", "--k_features", required=True)
args = parser.parse_args()

dataset_name = args.dataset
dataset_labels = dataset_name + "_target"
features = args.features

graph_features = load_graph_features(features)
print("Loaded %d features" % len(graph_features))
graph_dataset = load_all_graphs(dataset_name)
dataset_labels = load_labels(dataset_labels)

dataframe_name = dataset_name + "_" + features + "_flatten_df.dat"

# check if flatten dataframe exists
flatten_graph = os.path.join(os.path.dirname(__file__), dataframe_name)

if not os.path.exists(flatten_graph):
    df = {}
    for k, graph in enumerate(graph_dataset):
        print("%d/%d" % (k + 1, len(graph_dataset)))
        for i, subgraph in enumerate(graph_features):
            try:
                df[i].append(int(subgraph_exists(graph, subgraph)))
            except:
                df[i] = [int(subgraph_exists(graph, subgraph))]

    df["y"] = dataset_labels

    df = pd.DataFrame(df)

    save_features(flatten_graph, df)
    perform_selection(
        dataframe_name, df, args.selection, args.k_features, graph_features
    )
else:
    df = load_df(flatten_graph)
    perform_selection(
        dataframe_name, df, args.selection, args.k_features, graph_features
    )
