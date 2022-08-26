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
    
    
class GAT(torch.nn.Module):
    
    def __init__(self, num_node_features, num_classes, num_heads=8, hidden_dimensions=8, dropout=0, bias=True):
        super().__init__()
        
        self.gatconv1 = GATConv(num_node_features, hidden_dimensions, num_heads, bias=bias)
        self.gatconv2 = GATConv(hidden_dimensions*num_heads, num_classes, 1, bias=bias)
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = dropout_adj(edge_index, p=self.dropout)
        x = self.gatconv1(x, edge_index)
        x = F.elu(x)
        x = self.gatconv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
class SAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dimensions=64, bias=True):
        super().__init__()
        self.sageconv1 = SAGEConv(num_node_features, hidden_dimensions, bias=bias)
        self.sageconv2 = SAGEConv(hidden_dimensions, num_classes, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sageconv1(x, edge_index)
        x = F.relu(x)
        x = self.sageconv2(x, edge_index)
        return F.log_softmax(x, dim=1) 