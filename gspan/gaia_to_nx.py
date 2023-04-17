import argparse
import networkx as nx, os, pickle


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-nf", "--node_features", required=True)
args = parser.parse_args()

dataset = args.dataset
node_features = int(args.node_features) + 1
patterns_file = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data_gaia",
    dataset,
    "patternResult.txt",
)

ids = []
txt_lines = None
with open(patterns_file, "r") as file:
    txt_lines = file.read().splitlines()

for i, line in enumerate(txt_lines):
    if line.startswith("id"):
        ids.append(i)
ids.append(len(txt_lines))

features = []
len_ids = len(ids) - 1
for id in range(len_ids):
    adj = txt_lines[ids[id] + 2 : ids[id + 1] - 2]
    tmp_feature = []
    for line in adj:
        tmp_line = []
        for elem in line.split(" ")[:-1]:
            tmp_line.append(int(elem))
        tmp_feature.append(tmp_line)
    features.append(tmp_feature)


nx_graphs = []
for adj in features:
    m = len(adj)
    n = m

    graph = nx.Graph()

    for i in range(m):
        graph.add_node(i, label=str(adj[i][i] - 1))

    for i in range(m):
        for j in range(n):
            if i != j and adj[i][j] != 0:
                graph.add_edge(i, j, label=str(adj[i][j] - node_features))
    nx_graphs.append(graph)

data_write = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data_gaia",
    dataset,
    dataset + "_gaia_nx.dat",
)

with open(data_write, "wb") as write_file:
    pickle.dump(nx_graphs, write_file)
