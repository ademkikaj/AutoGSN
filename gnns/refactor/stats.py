import numpy as np
import json
from operator import itemgetter


def get_max(file):
    with open(file) as res:
        data = res.readlines()
        agg = []
        for line in data:
            line = json.loads(line)
            agg.append(line["Test Acc"])
        
        return max(agg)


dataset = "NCI1"
batch_size = 32
hidden_layers = 16
# data_trained/"+dataset+"/PROTEINS_834/
all_max = []
for i in range(1,11):
    all_max.append(get_max("results_"+dataset+"_"+str(i)+"_"+str(batch_size)+"_"+str(hidden_layers)+".log"))
print("%.2f±%.2f" % (np.average(all_max), np.std(all_max, ddof=1))) 

# print(max(res, key=itemgetter(1)))
# print("Batch: %d | %.2f, +-%.2f" % (32, avg, std))
