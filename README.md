import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import yfinance as yf
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import logging
from functools import wraps

# Настройка логирования
logging.basicConfig(filename='portfolio_optimization.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Декоратор для обработки ошибок
def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

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

@error_handler
def load_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

@error_handler
def create_stock_graph(data):
    corr_matrix = data.corr()
    graph = nx.from_pandas_adjacency(corr_matrix)
    return graph

@error_handler
def prepare_data_for_gnn(graph, data):
    node_features = torch.tensor(data.values, dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges())).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

@error_handler
def train_model(model, data, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

@error_handler
def optimize_portfolio(model, data, risk_tolerance):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
    weights = torch.softmax(predictions, dim=0).numpy()
    return weights

@error_handler
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

@error_handler
def backtest_strategy(data, weights, initial_capital=10000):
    portfolio_value = [initial_capital]
    returns = data.pct_change()
    for i in range(1, len(data)):
        daily_return = np.sum(returns.iloc[i] * weights)
        portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
    
    portfolio_returns = pd.Series(portfolio_value).pct_change()
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    max_drawdown = (pd.Series(portfolio_value) / pd.Series(portfolio_value).cummax() - 1).min()
    
    return {
        'final_value': portfolio_value[-1],
        'total_return': (portfolio_value[-1] / initial_capital - 1) * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

@error_handler
def visualize_portfolio_performance(data, weights, initial_capital=10000):
    portfolio_value = [initial_capital]
    returns = data.pct_change()
    for i in range(1, len(data)):
        daily_return = np.sum(returns.iloc[i] * weights)
        portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, portfolio_value)
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.savefig('portfolio_performance.png')
    plt.close()

@error_handler
def analyze_stock_data(data):
    returns = data.pct_change()
    corr_matrix = returns.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Stock Returns')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    stats = returns.describe()
    stats.loc['skew'] = returns.skew()
    stats.loc['kurtosis'] = returns.kurtosis()
    return stats

@error_handler
def forecast_stock_prices(data, steps=30):
    forecasts = {}
    for column in data.columns:
        model = ARIMA(data[column], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=steps)
        forecasts[column] = forecast
    return pd.DataFrame(forecasts)

def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    stock_data = load_stock_data(tickers, start_date, end_date)
    
    graph = create_stock_graph(stock_data)
    gnn_data = prepare_data_for_gnn(graph, stock_data)
    
    model = AdvancedStockGNN(num_features=stock_data.shape[1], hidden_channels=64, num_heads=8, num_classes=1)
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
    
    visualize_portfolio_performance(stock_data, optimal_weights)
    
    stats = analyze_stock_data(stock_data)
    print("\nStock statistics:")
    print(stats)
    
    future_prices = forecast_stock_prices(stock_data)
    print("\nForecasted prices for the next 30 days:")
    print(future_prices)

if __name__ == "__main__":
    main()
