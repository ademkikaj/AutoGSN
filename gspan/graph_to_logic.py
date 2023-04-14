import pickle
import networkx as nx

features_g = pickle.load(
    open("mutag_playground/MUTAG_20_top_5_xgb_selected_features.dat", "rb")
)
node_labels = ["zero", "one", "two", "three", "four", "five", "six"]
edge_labels = ["ezero", "eone", "etwo", "ethree"]

counter = 1
types = []
rmodes = []
features = []
for g in features_g:
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

for type in types:
    print(type)

for rmode in rmodes:
    print(rmode)
