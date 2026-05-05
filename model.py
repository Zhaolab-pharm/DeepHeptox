import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2, dropout=0.1):
        super().__init__()
        layer_list = []
        for i in range(layers-1):
            layer_list.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(dropout))
        layer_list.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layer_list)
    def forward(self, x):
        return self.layers(x)

class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, mlp_hidden=64, output_dim=1, heads=4, mlp_layers=2, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout)
        self.pool = global_mean_pool
        self.mlp = MLP(hidden_dim* heads, mlp_hidden, output_dim, mlp_layers, dropout)
    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.pool(x, batch)
        return self.mlp(x).squeeze(-1)
