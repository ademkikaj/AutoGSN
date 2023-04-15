import json, os, numpy as np, matplotlib.pyplot as plt, operator

experiments_dir = "experiments"
dataset_name = "MUTAG"

results = json.load(
    open(os.path.join(experiments_dir, dataset_name, "results", "MUTAG_20_top_10_xgb_selected_features_unlabelled_0.json"))
)

results = results["Train"][0]

all_epoch_test = {}
all_test_acc = []
all_train_acc = []

for fold in range(4):
    all_test_acc = []
    all_train_acc = []
    for epoch in range(0, 600):
        all_test_acc.append(float(results["Fold "+str(fold)][epoch]["Test Acc"]))
        all_train_acc.append(float(results["Fold "+str(fold)][epoch]["Train Acc"]))
        try:
            all_epoch_test[epoch] = all_epoch_test[epoch] + [float(results["Fold "+str(fold)][epoch]["Test Acc"])]
        except:
            all_epoch_test[epoch] = [float(results["Fold "+str(fold)][epoch]["Test Acc"])]
    plt.figure()
    plt.plot(all_test_acc)
    plt.plot(all_train_acc)
    plt.savefig("Fold "+str(fold))


epochs_avg_std = {}
for k, epoch in all_epoch_test.items():
    epochs_avg_std[k] = (np.average(epoch), np.std(epoch))
    
# print(epochs_avg_std)
print("Results Test Acc: %.2f±%.2f" % (max(epochs_avg_std.items(), key=operator.itemgetter(1))[1][0], max(epochs_avg_std.items(), key=operator.itemgetter(1))[1][1]))
print(max(epochs_avg_std.items(), key=operator.itemgetter(1)))
# lists = all_epoch_test.items()
# x, y = zip(*lists)
# plt.plot(x, y)




