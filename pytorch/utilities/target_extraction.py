import argparse, os
import pickle
from torch_geometric.datasets import TUDataset


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
args = parser.parse_args()

dataset = args.dataset
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset)
data_save = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data_networkx", dataset + "_target.dat"
)

dataset = TUDataset(root=data_root, name=dataset)

target = []
for data in dataset:
    target.append(int(data.y))

with open(data_save, "wb") as file:
    pickle.dump(target, file)
