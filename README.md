import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import yfinance as yf
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
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
    edge_index = torch.tensor(list(graph.edges())).t().contiguous()
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
    weights = torch.softmax(predictions, dim=0).numpy()
    return weights

def optimize_portfolio_markowitz(returns, target_return, risk_tolerance):
    n = returns.shape[1]
    def objective(weights):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return portfolio_volatility - risk_tolerance * portfolio_return

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n for _ in range(n)])
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def backtest_strategy(data, weights, initial_capital=10000):
    portfolio_value = [initial_capital]
    returns = data.pct_change()
    for i in range(1, len(data)):
        daily_return = np.sum(returns.iloc[i] * weights)
        portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
    return {
        'final_value': portfolio_value[-1],
        'total_return': (portfolio_value[-1] / initial_capital - 1) * 100,
        'sharpe_ratio': np.sqrt(252) * np.mean(returns) / np.std(returns),
        'max_drawdown': (np.maximum.accumulate(portfolio_value) - portfolio_value) / np.maximum.accumulate(portfolio_value)
    }

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
    
    print("Optimal portfolio weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")
    
    backtest_results = backtest_strategy(stock_data, optimal_weights)
    print("\nBacktest results:")
    for key, value in backtest_results.items():
        print(f"{key}: {value:.2f}")
