import os, argparse, json
from utilities.TrainDataset import TrainDataset

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
# parser.add_argument("-p", "--processed_dataset", required=True)
args = parser.parse_args()

experiments_dir = "experiments"
dataset_name = args.dataset

config_file = json.load(
    open(os.path.join(experiments_dir, dataset_name, "config.json"))
)

features = config_file["features"]
model = config_file["model"]

TrainDataset(
    model["seed"],
    dataset_name,
    "No_Features",
    False,
    model["folds"],
    model["epochs"],
    model["batch_size"],
    model["hidden_channel"],
    model["learning_rate"],
    model["layers"],
    "cuda:1",
    model["scheduler"],
    model["step_size"],
    model["decay"],
    model["dropout"],
)
