# AMsek24
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import yfinance as yf
import matplotlib.pyplot as plt

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
        def load_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def create_stock_graph(data):
    corr_matrix = data.corr()
    graph = nx.from_pandas_adjacency(corr_matrix)
    return graph
def prepare_data_for_gnn(graph, data):
    node_features = torch.tensor(data.values, dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

def train_model(model, data, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

def optimize_portfolio(model, data, risk_tolerance):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
    weights = torch.softmax(predictions, dim=0)
    return weights.numpy()

def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    stock_data = load_stock_data(tickers, start_date, end_date)

    graph = create_stock_graph(stock_data)
    gnn_data = prepare_data_for_gnn(graph, stock_data)

    model = StockGNN(num_features=stock_data.shape[1], hidden_channels=64, num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    train_model(model, gnn_data, optimizer, criterion, num_epochs=100)

    risk_tolerance = 0.5
    optimal_weights = optimize_portfolio(model, gnn_data, risk_tolerance)

    print("Оптимальные веса портфеля:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")
