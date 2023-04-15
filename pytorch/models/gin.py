import torch
from torch_geometric.nn import MLP, global_add_pool, Linear
import torch.nn.functional as F

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = MLP([in_channels, hidden_channels, hidden_channels])
        self.conv2 = MLP([hidden_channels, hidden_channels, hidden_channels])
        self.conv3 = MLP([hidden_channels, hidden_channels, hidden_channels])
        self.conv4 = MLP([hidden_channels, hidden_channels, hidden_channels])
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()
        h = self.conv4(h, edge_index)

        # Graph-level readout
        hG = global_add_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return hG, F.log_softmax(h, dim=1)