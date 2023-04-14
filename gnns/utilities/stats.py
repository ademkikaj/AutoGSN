import numpy as np
import json
from operator import itemgetter


def get_max(file):
    with open(file) as res:
        data = json.load(res)
        epochs = data["EPOCHS"]
        counter = 1
        agg = []
        res = []
        for epoch in epochs:
            agg.append(epoch["Test Acc"])
            if (counter % 10) == 0:
                avg = np.average(agg)
                std = np.std(agg, ddof=1)
                res.append((avg, std))
                agg = []
            counter += 1
        return res


res = get_max("results_PROTEINS_3.log")

print(max(res, key=itemgetter(1)))

# print("Batch: %d | %.2f, +-%.2f" % (32, avg, std))
