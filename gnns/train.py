import os, argparse, json
from multiprocessing import Process, set_start_method
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

if __name__ == "__main__":
    set_start_method("spawn")
    procs = []
    for feature in features:
        proc: Process = Process(
            target=TrainDataset,
            args=(
                model["seed"],
                dataset_name,
                feature["name"],
                feature["labelled"],
                model["folds"],
                model["epochs"],
                model["batch_size"],
                model["hidden_channel"],
                model["learning_rate"],
                model["layers"],
                feature["device"],
                model["scheduler"],
                model["step_size"],
                model["decay"],
                model["dropout"],
            ),
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
