import torch, os
import torch_geometric
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import HeteroData


class HeteroGNN(torch.nn.Module):
    def __init__(
        self, hidden_channels, num_classes, num_layers, node_types, edge_types
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {key: SAGEConv(-1, hidden_channels) for key in edge_types}, aggr="sum"
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, num_classes)
        self.node_types = node_types

    def forward(self, x_dict, edge_index_dict, batch):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        x = x_dict[self.node_types[0]]
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return torch.squeeze(self.lin(x))


dataset = "MUTAG"
data_root = os.path.join(os.path.dirname(__file__), "data", dataset)
dataset = TUDataset(root=data_root, name=dataset)
print(dataset)
# print(len(dataset))

mutag_hetero = HeteroData()
counter = 0
for graph in dataset:
    print(graph)
    print(
        graph.to_heterogeneous(
            node_type_names=["Nodes"],
            edge_type=torch.argmax(graph.edge_attr, dim=1),
            edge_type_names=[
                ("nodes", "edge%d" % d, "nodes")
                for d in range(int(torch.max(graph.edge_attr)) + 1)
            ],
        )
    )

    # print(graph.edge_index)
    # print(graph.edge_attr)

    # nodes
    node_types = {}
    for d in range(torch.max(torch.argmax(graph.x, dim=1)) + 1):
        node_types[d] = []
    all_nodes = graph.x
    for i, node in enumerate(all_nodes):
        index = int((node == 1).nonzero(as_tuple=False)[0])
        node_types[index].append(i)

    for node_key, node_value in node_types.items():
        mutag_hetero["node_%d" % node_key].x = torch.tensor(node_value)
    # edges
    u_edges = graph.edge_index[0]
    v_edges = graph.edge_index[1]
    len_edges = len(u_edges)
    all_edges = {}
    for i in range(len_edges):
        u = u_edges[i]
        v = v_edges[i]
        u_node_type = (graph.x[u] == 1).nonzero(as_tuple=False)[0]
        v_node_type = (graph.x[v] == 1).nonzero(as_tuple=False)[0]
        u_v_edge_type = (graph.edge_attr[i] == 1).nonzero(as_tuple=False)[0]
        try:
            all_edges[
                "node_%d, edge_%d, node_%d"
                % (int(u_node_type[0]), int(u_v_edge_type[0]), int(v_node_type[0]))
            ] = torch.cat(
                (
                    all_edges[
                        "node_%d, edge_%d, node_%d"
                        % (
                            int(u_node_type[0]),
                            int(u_v_edge_type[0]),
                            int(v_node_type[0]),
                        )
                    ],
                    torch.tensor([[int(u)], [int(v)]]),
                ),
                1,
            )
        except:
            all_edges[
                "node_%d, edge_%d, node_%d"
                % (int(u_node_type[0]), int(u_v_edge_type[0]), int(v_node_type[0]))
            ] = torch.tensor([[int(u)], [int(v)]])

    for k, v in all_edges.items():
        mutag_hetero[k].edge_index = v

    mutag_hetero.y = graph.y
    print(mutag_hetero)
    exit()
    # edge types

    # edge_attr
