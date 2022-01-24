import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_adj


class MLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False):
        
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )
        
        
    def forward(self, x):
        
        h = self.model(x)
        return h

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dimension=64, dropout_prob=0, bias=True):
        
        super().__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_dimension, cached=True, bias=bias)
        self.conv2 = GCNConv(hidden_dimensions, num_classes, cached=True, bias=bias)
        self.dropout_prob = dropout_prob

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        if self.dropout_prob > 0:
            edge_index, _ = dropout_adj(edge_index, p=self.dropout_prob)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)