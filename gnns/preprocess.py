import os, argparse, json
from multiprocessing import Process
from utilities.ProcessData import ProcessData


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
args = parser.parse_args()

experiments_dir = "experiments"
dataset_name = args.dataset

config_file = json.load(
    open(os.path.join(experiments_dir, dataset_name, "config.json"))
)

features = config_file["features"]

if __name__ == "__main__":
    # preprocess
    procs = []
    for feature in features:
        proc: Process = Process(
            target=ProcessData,
            args=(dataset_name, feature["name"], feature["labelled"]),
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
