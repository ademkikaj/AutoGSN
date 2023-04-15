import torch
from torch_geometric.nn import GIN, GINConv, MLP, Linear, global_add_pool
import torch.nn.functional as F


class GINDefault(torch.nn.Module):
    """GCN"""

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super(GINDefault, self).__init__()
        self.gin = GIN(in_channels, hidden_channels, num_layers)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.gin(x, edge_index)

        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)

        return out


class GINPaper(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super(GINPaper, self).__init__()
        self.gnn_layers = torch.nn.ModuleList()
        self.output_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP(
                [hidden_channels if i > 0 else in_channels, hidden_channels],
                batch_norm=True,
            )
            gin = GINConv(mlp, train_eps=False)
            self.gnn_layers.append(gin)

        for i in range(num_layers):
            self.output_layers.append(Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        hidden_layers = []
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            hidden_layers.append(x)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_layers):
            pooled_h = global_add_pool(h, batch)
            output_hidden_layer = self.output_layers[layer](pooled_h)
            score_over_layer += F.dropout(
                output_hidden_layer, 0.5, training=self.training
            )

        return score_over_layer
