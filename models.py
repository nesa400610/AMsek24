import torch
from torch_geometric.nn import GCNConv, GATConv

class StockGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(StockGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class AdvancedStockGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, num_classes):
        super(AdvancedStockGNN, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = torch.nn.functional.elu(self.conv1(x, edge_index))
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
