import torch
from torch_geometric.nn import GIN, GINConv, MLP, Linear, global_add_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d


class GINDefault(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, out, dropout):
        super(GINDefault, self).__init__()
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            gin = GIN(hidden_channels if i > 0 else in_channels, 
                      hidden_channels, 2, 
                      norm=BatchNorm1d([hidden_channels if i > 0 else in_channels, hidden_channels]))
            self.gnn_layers.append(gin)
        self.lin = Linear(hidden_channels, out)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        hidden_layers = []
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            hidden_layers.append(x)
            
        score_over_layer = 0
        for layer, h in enumerate(hidden_layers):
            pooled_h = global_add_pool(h, batch)
            output_hidden_layer = self.output_layers[layer](pooled_h)
            score_over_layer += F.dropout(
                output_hidden_layer, self.dropout, training=self.training
            )
        # out = self.lin(x)

        return score_over_layer