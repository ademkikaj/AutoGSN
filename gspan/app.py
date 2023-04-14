from collections import Counter, defaultdict
import os
from gspan_mining.config import parser
import networkx as nx
from gspan_mining import gSpan
import matplotlib.pyplot as plt
from gspan_mining.main import main
import pickle


def plot_graph(graph1: nx.Graph):
    pos = nx.spring_layout(graph1)
    nx.draw_networkx_nodes(graph1, pos, graph1.nodes)
    nx.draw_networkx_edges(graph1, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph1, pos, labels=nx.get_node_attributes(graph1, "label"))
    nx.draw_networkx_edge_labels(
        graph1, pos, edge_labels=nx.get_edge_attributes(graph1, "label")
    )
    plt.show()


# args_str = "-s 170 -l 5 data_graph/MUTAG.graph"
# FLAGS, _ = parser.parse_known_args(args=args_str.split())
# print(FLAGS)
# exit()
# gs = main(FLAGS)

# exit()

dataset = "NCI1_3000"
graph = os.path.join("data_graph", dataset + ".graph")
support = int(4110 * 0.9)
print(support)
lower_bound = 2
upper_bound = float("inf")
max_ngraph = float("inf")
is_undirected = True
verbose = False
visualize = False
where = False

gs = gSpan(
    database_file_name=graph,
    min_support=support,
    min_num_vertices=lower_bound,
    max_num_vertices=upper_bound,
    max_ngraphs=max_ngraph,
    is_undirected=is_undirected,
    verbose=verbose,
    visualize=visualize,
    where=where,
)
nx_graphs = []
gs._read_graphs()
for i, g in gs.graphs.items():
    nx_graphs.append(g.to_nx_graph())

# print(gs._min_num_vertices)

data_write = os.path.join(
    os.path.dirname(__file__),
    "data_nx_features",
    dataset + "_" + str(support) + ".dat",
)

# gs.run()
# gs.time_stats()
# node_labels = ["zero", "one", "two", "three", "four", "five", "six"]
# edge_labels = ["ezero", "eone", "etwo", "ethree"]

# for graph in gs.nx_graphs:
#     plot_graph(graph)
# exit()

# print(len(gs.nx_graphs))

# for g_i in gs.nx_graphs:
#     for g_j in gs.nx_graphs:
#         if nx.is_isomorphic(g_i, g_j) and len(g_i.nodes) > 5:
#             print(g_i)
#             print(g_j)
#             plot_graph(g_i)
#             plot_graph(g_j)
#             exit()

# exit()
print(len(nx_graphs))
plot_graph(nx_graphs[0])
with open(data_write, "wb") as write_file:
    pickle.dump(nx_graphs, write_file)

exit()

counter = 1
types = []
rmodes = []
features = []
hashed_list = defaultdict(list)
for g in gs.nx_graphs:
    feature = []
    curr_nx_graph: nx.Graph = g
    args_s = "(%s)" % (", ".join([chr(65 + i) for i in curr_nx_graph.nodes]))
    args_type_s = "(%s)" % (", ".join(["obj" for i in curr_nx_graph.nodes]))
    args_rmode_s = "(%s)" % (", ".join(["+-S%d" % i for i in curr_nx_graph.nodes]))
    feature_s = "feature_%d%s :- " % (counter, args_s)
    type_s = "type(feature_%d%s)." % (counter, args_type_s)
    rmode_s = "rmode(feature_%d%s)." % (counter, args_rmode_s)
    types.append(type_s)
    rmodes.append(rmode_s)
    feature.append(feature_s)
    for node in curr_nx_graph.nodes:
        feature.append(
            node_labels[int(curr_nx_graph.nodes[node]["label"])]
            + "(%s)" % chr(65 + node)
        )
    for edge in curr_nx_graph.edges:
        u = edge[0]
        v = edge[1]
        feature.append(
            "%s(%s,%s)"
            % (
                edge_labels[int(curr_nx_graph.get_edge_data(u, v)["label"])],
                chr(65 + u),
                chr(65 + v),
            )
        )

    hashed_list[
        nx.weisfeiler_lehman_graph_hash(
            curr_nx_graph, edge_attr="label", node_attr="label"
        )
    ].append(curr_nx_graph)
    features.append(feature)
    counter += 1

for feature in features:
    for k, element in enumerate(feature):
        if k == 0:
            print(element, end=" ")
        elif k == len(feature) - 1:
            print(element, end=".")
        else:
            print(element, end=",")
    print()

# for type in types:
#     print(type)

# for rmode in rmodes:
#     print(rmode)
# print(gs._frequent_subgraphs)

# print(gs._frequent_subgraphs[0])

# print(Counter(hashed_list))


def count_iso(list_of_graphs):
    counter = 0
    black_list = set()
    for i in range(len(list_of_graphs)):
        for j in range(i, len(list_of_graphs)):
            if j not in black_list:
                res = nx.is_isomorphic(list_of_graphs[i], list_of_graphs[j])
                if res == False:
                    print("HERE")
                    nx.draw_networkx(list_of_graphs[i])
                    plt.savefig("1.png")
                    nx.draw_networkx(list_of_graphs[j])
                    plt.savefig("2.png")
                counter += res
                if res:
                    black_list.add(j)
    return len(list_of_graphs) - len(black_list)


hashed_list_iso = {}
for k, graphs in hashed_list.items():
    hashed_list_iso[k] = count_iso(graphs)

# for k, v in hashed_list.items():
#     print(k, len(v), hashed_list_iso[k])
