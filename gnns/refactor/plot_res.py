import matplotlib.pyplot as plt
import pickle


data_name = "IMDB-BINARY"

train_acc = pickle.load(open("train_acc_gin.dat", "rb"))
# print(train_acc)
plt.plot(train_acc)
plt.ylim([0, 1])
plt.show()
exit()


def plot():
    plt.title("IMDB_BINARY")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Acc")
    plt.legend()
    plt.show()


def single():
    for i in range(3, 10):
        res = "res_" + data_name + "_" + str(i) + ".dat"
        scores = pickle.load(open(res, "rb"))
        plt.plot(scores, label=i)


def double():
    for i in range(3, 9):
        res = "res_" + data_name + str(i) + "_" + str(i + 1) + ".dat"
        scores = pickle.load(open(res, "rb"))
        plt.plot(scores, label=str(i) + "_" + str(i + 1))


def increase():
    for i in range(5, 10, 2):
        res = "res_" + data_name + "3_" + str(i) + ".dat"
        scores = pickle.load(open(res, "rb"))
        plt.plot(scores, label="3_" + str(i))


# single()
# plot()
# double()
# plot()
# increase()
# plot()
