import argparse
import os
from gspan_mining import gSpan
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-s", "--support", required=True)
parser.add_argument("-n", "--nr_graphs", required=True)
args = parser.parse_args()

data = args.dataset

dataset = args.dataset
graph = os.path.join("data_graph", dataset + ".graph")
support = int(int(args.nr_graphs) * (int(args.support) / 100))
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

data_write = os.path.join(
    os.path.dirname(__file__),
    "data_nx_features",
    dataset + "_" + str(support) + ".dat",
)

gs.run()
gs.time_stats()

print("Number of generated subgraphs: %d" % len(gs.nx_graphs))
with open(data_write, "wb") as write_file:
    pickle.dump(gs.nx_graphs, write_file)

