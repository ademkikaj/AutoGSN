import torch
from torch_geometric.data import Data
import pickle, collections

dataset = "IMDB-BINARY"
file_name = "processed" + dataset + ".dat"

a = torch.empty(0, 1)

b = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])

lst = [7, 8]
lst_tensor = torch.empty(3, 1)
for i, v in enumerate(lst):
    lst_tensor[i] = v
    # lst_tensor = torch.cat((lst_tensor, torch.FloatTensor([[e]])), 0)

print(lst_tensor)


# c = torch.cat((b, torch.FloatTensor([[6, 1]])), 0)
# a = torch.cat((a, torch.FloatTensor([[7]])), 0)
# print(torch.cat((a, torch.FloatTensor([[1, 1]])), 0))
print(torch.cat((b, lst_tensor), 1))
# print(b)

# print(torch.cat((b, a), 1))


exit()

print(b.shape)
z = torch.rand((2,))
print(z)
print(torch.cat((b, z), 1))


print(b.shape)
print(list_to_tensor.shape)

print(b.reshape_as(list_to_tensor))
# print(b.expand(1))
# print(b.expand(2))
exit()

for t in b:
    print(t.unsqueeze(0))
    print(t.unsqueeze(0).expand(3))
    torch.cat((t.unsqueeze(0), list_to_tensor), 0)
    print(t)
cat = torch.cat((b, list_to_tensor), 1)
print(cat)
exit()

print(a[:1])

for i in range(1, 2):
    print(i)

all_features = torch.empty(0, 2)

a = torch.FloatTensor([[1, 2]])
print(a)
b = torch.tensor([[3, 4]])
c = torch.tensor([[5, 6]])

all_features = torch.cat((all_features, a), 0)
print(all_features)
all_features = torch.cat((all_features, b), 0)
print(all_features)
all_features = torch.cat((all_features, c), 0)

print(all_features)
exit()

proccesed_load = pickle.load(open(file_name, "rb"))
print(proccesed_load[0])
a = proccesed_load[0]
print(a)
print(max(a))
print(min(a))
# for i in range(1, len(proccesed_load[0])):
#     print(proccesed_load[0][i])
#     print(list(proccesed_load[0][i].values()))

# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)

# x = torch.tensor([[-1,2], [-5,4], [3,5], [1,1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index.t().contiguous())

# print(data)

# d_features = {0: {1: 2, 2: 1},
#               1: {1: 20, 2: 15},
#               2: {1: 8, 2:3}}

# t_features = torch.empty(3,2)
# print(t_features)
# print(t_features[0])

# for k,v in d_features.items():
#     t_features[k] = torch.FloatTensor(list(v.values()))
#     print(t_features)


# data.x = torch.cat((x, t_features), 1)

# for k,v in enumerate(data.x):
#     print(v)

# for node_features in data.x:
#     print(node_features)

# data.x[0] = torch.tensor([3,18])

# for node_features in data.x:
#     print(node_features)
# print(data.x[0])
