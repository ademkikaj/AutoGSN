import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.feature_selection as fs

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel, SelectKBest


def pickle_features(dataset, features):
    selected_features = os.path.join(
        os.path.dirname(__file__), dataset + "_selected_features.dat"
    )
    pickle.dump(features, open(selected_features, "wb"))


def load_features(dataset):
    features_graph = os.path.join(
        os.path.dirname(__file__),
        dataset + ".dat",
    )

    return pickle.load(open(features_graph, "rb"))


def load_df(dataset):
    features_df = os.path.join(
        os.path.dirname(__file__),
        dataset + "_df.dat",
    )

    return pickle.load(open(features_df, "rb"))


def plot_graphs(graph_pos: nx.Graph, graph_neg: nx.Graph):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    pos = nx.spring_layout(graph_pos, seed=1, iterations=100)
    nx.draw_networkx_nodes(graph_pos, pos, graph_pos.nodes, ax=ax1)
    nx.draw_networkx_edges(graph_pos, pos, width=1.0, alpha=0.5, ax=ax1)
    nx.draw_networkx_labels(
        graph_pos, pos, labels=nx.get_node_attributes(graph_pos, "label"), ax=ax1
    )
    nx.draw_networkx_edge_labels(
        graph_pos, pos, edge_labels=nx.get_edge_attributes(graph_pos, "label"), ax=ax1
    )

    pos = nx.spring_layout(graph_neg, seed=1, iterations=100)
    nx.draw_networkx_nodes(graph_neg, pos, graph_neg.nodes, ax=ax2)
    nx.draw_networkx_edges(graph_neg, pos, width=1.0, alpha=0.5, ax=ax2)
    nx.draw_networkx_labels(
        graph_neg, pos, labels=nx.get_node_attributes(graph_neg, "label"), ax=ax2
    )
    nx.draw_networkx_edge_labels(
        graph_neg, pos, edge_labels=nx.get_edge_attributes(graph_neg, "label"), ax=ax2
    )

    plt.show()


def plot_graph(graph: nx.Graph):
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx_nodes(graph, pos, graph.nodes)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, "label"))
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=nx.get_edge_attributes(graph, "label")
    )

    plt.show()


def chi_square_features(train, labels, top_k, features_graph):
    chi_scores = fs.chi2(train, labels)
    p_values = pd.Series(chi_scores[1], index=train.columns)
    p_values.sort_values(ascending=True, inplace=True)
    p_values = p_values[-top_k:]
    selected_graphs = []
    for key in p_values.keys():
        print(key)
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
    plot_importance(model)
    plt.show()
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    feature_important = model.get_booster().get_score(importance_type="weight")
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(
        by="score", ascending=False
    )
    print(data.head(top_k))
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
        print(key)
        selected_graphs.append(features_graph[key])
    return selected_graphs

dataset = "MUTAG_20"

df: pd.DataFrame = load_df(dataset)

features_graph = load_features(dataset)

train = df.drop("y", axis=1)
label = df["y"]

res = xgb_features(train, label, 20, features_graph)
# res = random_forest_features(train, label, 10, features_graph)
# res = chi_square_features(train, label, 20, features_graph)
# res = mutual_info(train, label, 5, features_graph)
# for g in res:
# plot_graph(g)
pickle_features(dataset + "_top_20_xgb", res)
