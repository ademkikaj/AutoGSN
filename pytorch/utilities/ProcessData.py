from collections import Counter
import pickle, networkx as nx
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
import shutil, time, json, os


class ProcessData:
    def __init__(self, dataset_name, feature_name, labelled=False) -> None:
        self.dataset_name = dataset_name
        self.feature_name = feature_name
        self.labelled = labelled
        self.graph_index = 0
        self.processed_dataset = None
        self.results = {}
        self.processed_data_dir = os.path.join(
            "experiments",
            self.dataset_name,
            "data",
            "processed",
            self.feature_name + ("_labelled" if self.labelled else "_unlabelled"),
        )
        self.nx_graphs = pickle.load(
            open(
                os.path.join(
                    "experiments", dataset_name, "data", "nx", dataset_name + ".dat"
                ),
                "rb",
            )
        )
        self.features = pickle.load(
            open(
                os.path.join(
                    "experiments", dataset_name, "features", feature_name + ".dat"
                ),
                "rb",
            )
        )
        self._load()
        self._move_processed_data()
        self._clean_raw()

    def _load(self):
        self.results["feature_name"] = self.feature_name
        self.results["nr_of_features"] = len(self.features)
        start_time = time.time()
        self.processed_dataset = TUDataset(
            root=self.processed_data_dir.__str__(),
            name=self.dataset_name,
            pre_transform=self._process,
        )
        self.results["elapsed_time"] = time.time() - start_time

    def _move_processed_data(self):
        shutil.move(self.processed_dataset.processed_dir, self.processed_data_dir)
        with open(os.path.join(self.processed_data_dir, "stats.json"), "w") as file:
            file.write(json.dumps(self.results))

    def _clean_raw(self):
        shutil.rmtree(os.path.join(self.processed_data_dir, self.dataset_name))

    def _plot(self, graph: nx.Graph, feature: nx.Graph):
        print(graph.edges)
        plt.figure(1)
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)

        edge_labels = {}
        for edge in graph.edges:
            edge_labels[(edge[0], edge[1])] = graph.get_edge_data(edge[0], edge[1])[
                "label"
            ]

        nx.draw_networkx_edge_labels(graph, pos, edge_labels)

        plt.figure(2)
        pos = nx.spring_layout(feature)
        nx.draw(feature, pos)
        nx.draw_networkx_nodes(feature, pos)
        nx.draw_networkx_edges(feature, pos)
        nx.draw_networkx_labels(feature, pos)
        nx.draw_networkx_edge_labels(feature, pos)

    def _process(self, graph) -> TUDataset:
        nx_graph: nx.Graph = self.nx_graphs[self.graph_index]
        for feature in self.features:
            counter = Counter()
            if self.labelled:
                subgraphs = nx.isomorphism.GraphMatcher(
                    nx_graph,
                    feature,
                    node_match=nx.isomorphism.categorical_node_match("label", None),
                    edge_match=nx.isomorphism.categorical_edge_match("label", None),
                ).subgraph_isomorphisms_iter()
            else:
                subgraphs = nx.isomorphism.GraphMatcher(
                    nx_graph, feature
                ).subgraph_isomorphisms_iter()
            subgraphs = [tuple((subgraph.keys())) for subgraph in subgraphs]
            cleaned = set(tuple(sorted(l)) for l in subgraphs)
            unpacked = []
            for k in cleaned:
                unpacked.append(list(k))
            counter = Counter()
            unpacked = [item for sublist in unpacked for item in sublist]
            counter.update(unpacked)
            feature_tensor = torch.zeros(nx_graph.number_of_nodes(), 1)
            for k, v in counter.items():
                feature_tensor[k] = v
            graph.x = torch.cat((graph.x, feature_tensor), 1)
        self.graph_index += 1
        return graph

    def _process_add_node(self, graph) -> TUDataset:
        nx_graph: nx.Graph = self.nx_graphs[self.graph_index]
        for feature in self.features:
            counter = Counter()
            if self.labelled:
                subgraphs = nx.isomorphism.GraphMatcher(
                    nx_graph,
                    feature,
                    node_match=nx.isomorphism.categorical_node_match("label", None),
                    edge_match=nx.isomorphism.categorical_edge_match("label", None),
                ).subgraph_isomorphisms_iter()
            else:
                subgraphs = nx.isomorphism.GraphMatcher(
                    nx_graph, feature
                ).subgraph_isomorphisms_iter()
            subgraphs = [tuple((subgraph.keys())) for subgraph in subgraphs]
            cleaned = set(tuple(sorted(l)) for l in subgraphs)
            unpacked = []
            for k in cleaned:
                unpacked.append(list(k))
            counter = Counter()
            unpacked = [item for sublist in unpacked for item in sublist]
            counter.update(unpacked)

            if len(counter.items()) > 0:            
                nodes_to_add = counter.most_common(1)[0][1]
                for i in range(0,nodes_to_add):
                    graph.x = torch.cat((graph.x, torch.FloatTensor([[0]*graph.x.shape[1]])), 0)
                    
                for element in counter.items():
                    node = element[0]
                    to_link = element[1]
                    for i in range(0,to_link):
                        new_edge = torch.IntTensor([[node],[(len(graph.x)-i)-1]])
                        graph.edge_index = torch.cat((graph.edge_index, new_edge), 1)
                        graph.edge_attr = torch.cat((graph.edge_attr, torch.FloatTensor([[0]*graph.edge_attr.shape[1]])), 0)
        self.graph_index += 1
        return graph
    
    def _process_edge(self, graph) -> TUDataset:
        nx_graph: nx.Graph = self.nx_graphs[self.graph_index]
        for feature in self.features:
            if self.labelled:
                subgraphs = nx.isomorphism.GraphMatcher(
                    nx_graph,
                    feature,
                    node_match=nx.isomorphism.categorical_node_match("label", None),
                    edge_match=nx.isomorphism.categorical_edge_match("label", None),
                ).subgraph_isomorphisms_iter()
            else:
                subgraphs = nx.isomorphism.GraphMatcher(
                    nx_graph, feature
                ).subgraph_isomorphisms_iter()

            subgraphs = [tuple((subgraph.keys())) for subgraph in subgraphs]
            cleaned = set(tuple(sorted(l)) for l in subgraphs)
            unpacked = []
            for k in cleaned:
                unpacked.append(list(k))
            counter = Counter()
            unpacked = [item for sublist in unpacked for item in sublist]
            counter.update(unpacked)

            for edge in nx_graph.edges:
                old_data = list(nx_graph.get_edge_data(edge[0], edge[1])["edge_attr"])
                old_data.append(0.0)
                nx_graph[edge[0]][edge[1]].clear()
                nx_graph.add_edge(edge[0], edge[1], edge_attr=old_data)

            for items in cleaned:
                for node in items:
                    neighbors = nx_graph.neighbors(node)
                    for neighbour in neighbors:
                        if neighbour in items and neighbour > node:
                            old_data = list(
                                nx_graph.get_edge_data(node, neighbour)["edge_attr"]
                            )
                            old_data[len(old_data) - 1] = (
                                old_data[len(old_data) - 1] + 1
                            )
                            nx_graph.add_edge(node, neighbour, edge_attr=old_data)
        for node in nx_graph.nodes:
            del nx_graph.nodes[node]["label"]
        graph_processed = torch_geometric.utils.from_networkx(nx_graph)
        graph_processed.y = graph.y
        self.graph_index += 1
        return graph_processed
    