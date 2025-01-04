import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import yfinance as yf
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import logging
from functools import wraps
import time
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import norm
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

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

# Декоратор для измерения времени выполнения функции
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
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

class SAGEStockGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SAGEStockGNN, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

@error_handler
@timing_decorator
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
@timing_decorator
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
@timing_decorator
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
@timing_decorator
def forecast_stock_prices(data, steps=30):
    forecasts = {}
    for column in data.columns:
        model = ARIMA(data[column], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=steps)
        forecasts[column] = forecast
    return pd.DataFrame(forecasts)

@error_handler
def calculate_portfolio_var(weights, returns, confidence_level=0.95):
    portfolio_returns = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    var = norm.ppf(1 - confidence_level) * portfolio_volatility * np.sqrt(1/252)
    return -var

@error_handler
def calculate_portfolio_cvar(weights, returns, confidence_level=0.95):
    var = calculate_portfolio_var(weights, returns, confidence_level)
    cvar = -np.mean(returns[returns < var])
    return cvar

@error_handler
def optimize_portfolio_mean_cvar(returns, target_return, confidence_level=0.95):
    n = returns.shape[1]
    def objective(weights):
        return calculate_portfolio_cvar(weights, returns, confidence_level)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n for _ in range(n)])
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

@error_handler
def calculate_portfolio_metrics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    var = calculate_portfolio_var(weights, returns)
    cvar = calculate_portfolio_cvar(weights, returns)
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'var': var,
        'cvar': cvar
    }

@error_handler
def visualize_efficient_frontier(returns, num_portfolios=1000):
    results = []
    for _ in range(num_portfolios):
        weights = np.random.random(returns.shape[1])
        weights /= np.sum(weights)
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        results.append([portfolio_return, portfolio_volatility, weights])
    
    results = np.array(results)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results[:, 1], results[:, 0], c=results[:, 0] / results[:, 1], marker='o')
    plt.colorbar(label='Sharpe ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.savefig('efficient_frontier.png')
    plt.close()

@error_handler
def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

@error_handler
def calculate_alpha(stock_returns, market_returns, risk_free_rate):
    beta = calculate_beta(stock_returns, market_returns)
    alpha = np.mean(stock_returns) - risk_free_rate - beta * (np.mean(market_returns) - risk_free_rate)
    return alpha

@error_handler
def calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate):
    beta = calculate_beta(portfolio_returns, market_returns)
    return (np.mean(portfolio_returns) - risk_free_rate) / beta

@error_handler
def calculate_information_ratio(portfolio_returns, benchmark_returns):
    active_return = portfolio_returns - benchmark_returns
    tracking_error = np.std(active_return)
    return np.mean(active_return) / tracking_error

@error_handler
def calculate_sortino_ratio(returns, risk_free_rate, target_return=0):
    downside_returns = returns[returns < target_return]
    downside_deviation = np.std(downside_returns)
    excess_return = np.mean(returns) - risk_free_rate
    return excess_return / downside_deviation

@error_handler
def calculate_maximum_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

@error_handler
def calculate_calmar_ratio(returns, risk_free_rate):
    total_return = (1 + returns).prod() - 1
    max_drawdown = calculate_maximum_drawdown(returns)
    return (total_return - risk_free_rate) / abs(max_drawdown)

@error_handler
def calculate_omega_ratio(returns, threshold=0):
    return_threshold = returns - threshold
    positive_returns = return_threshold[return_threshold > 0].sum()
    negative_returns = abs(return_threshold[return_threshold < 0].sum())
    return positive_returns / negative_returns

@error_handler
def calculate_kappa_three_ratio(returns, risk_free_rate, target_return=0):
    excess_return = returns - risk_free_rate
    downside_deviation = np.sqrt(np.mean(np.minimum(excess_return - target_return, 0)**3))
    return (np.mean(excess_return) - target_return) / downside_deviation

@error_handler
def calculate_upside_potential_ratio(returns, target_return=0):
    upside_returns = returns[returns > target_return]
    downside_returns = returns[returns < target_return]
    upside_potential = np.mean(upside_returns - target_return)
    downside_risk = np.sqrt(np.mean(downside_returns**2))
    return upside_potential / downside_
@error_handler
def calculate_upside_potential_ratio(returns, target_return=0):
    upside_returns = returns[returns > target_return]
    downside_returns = returns[returns < target_return]
    upside_potential = np.mean(upside_returns - target_return)
    downside_risk = np.sqrt(np.mean(downside_returns**2))
    return upside_potential / downside_risk

@error_handler
def calculate_pain_ratio(returns, risk_free_rate):
    excess_return = np.mean(returns) - risk_free_rate
    pain_index = np.mean(np.abs(np.maximum(0, np.maximum.accumulate(returns) - returns)))
    return excess_return / pain_index
@error_handler
def calculate_gain_loss_ratio(returns):
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    return np.mean(gains) / abs(np.mean(losses))
@error_handler
def calculate_ulcer_index(returns):
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = 1 - cumulative_returns / np.maximum.accumulate(cumulative_returns)
    return np.sqrt(np.mean(drawdowns**2))

@error_handler
def calculate_martin_ratio(returns, risk_free_rate):
    excess_return = np.mean(returns) - risk_free_rate
    ulcer_index = calculate_ulcer_index(returns)
    return excess_return / ulcer_index

@error_handler
def calculate_burke_ratio(returns, risk_free_rate):
    excess_return = np.mean(returns) - risk_free_rate
    drawdowns = 1 - (1 + returns).cumprod() / np.maximum.accumulate((1 + returns).cumprod())
    return excess_return / np.sqrt(np.sum(drawdowns**2))

@error_handler
def calculate_sterling_ratio(returns, risk_free_rate):
    excess_return = np.mean(returns) - risk_free_rate
    max_drawdown = calculate_maximum_drawdown(returns)
    return excess_return / max_drawdown
