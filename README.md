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
@error_handler
def calculate_portfolio_turnover(weights_before, weights_after):
    return np.sum(np.abs(weights_after - weights_before)) / 2
@error_handler
def calculate_herfindahl_index(weights):
    return np.sum(weights**2)

@error_handler
def calculate_diversification_ratio(weights, covariance_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    weighted_volatilities = np.sum(weights * np.sqrt(np.diag(covariance_matrix)))
    return weighted_volatilities / np.sqrt(portfolio_variance)
@error_handler
def calculate_information_ratio(returns, benchmark_returns):
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    return np.mean(active_returns) / tracking_error

@error_handler
def calculate_market_neutral_beta(returns, market_returns):
    market_neutral_returns = returns - market_returns
    beta = calculate_beta(market_neutral_returns, market_returns)
    return beta

@error_handler
def calculate_downside_deviation(returns, target_return=0):
    downside_returns = returns[returns < target_return]
    return np.sqrt(np.mean((downside_returns - target_return)**2))

@error_handler
def calculate_upside_deviation(returns, target_return=0):
    upside_returns = returns[returns > target_return]
    return np.sqrt(np.mean((upside_returns - target_return)**2))
@error_handler
def calculate_tail_ratio(returns, percentile=5):
    left_tail = np.percentile(returns, percentile)
    right_tail = np.percentile(returns, 100 - percentile)
    return abs(right_tail) / abs(left_tail)
@error_handler
def calculate_value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

@error_handler
def calculate_conditional_value_at_risk(returns, confidence_level=0.95):
    var = calculate_value_at_risk(returns, confidence_level)
    return np.mean(returns[returns <= var])
@error_handler
def calculate_risk_parity_weights(covariance_matrix):
    n = covariance_matrix.shape[0]
    def risk_parity_objective(weights):
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        risk_contributions = weights * (np.dot(covariance_matrix, weights)) / portfolio_risk
        return np.sum((risk_contributions - risk_contributions.mean())**2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n for _ in range(n)])
    result = minimize(risk_parity_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

@error_handler
def calculate_factor_exposures(returns, factor_returns):
    factor_model = sm.OLS(returns, sm.add_constant(factor_returns)).fit()
    return factor_model.params[1:]
@error_handler
def calculate_tracking_error(returns, benchmark_returns):
    active_returns = returns - benchmark_returns
    return np.std(active_returns) * np.sqrt(252)

@error_handler
def calculate_information_coefficient(predicted_returns, actual_returns):
    return np.corrcoef(predicted_returns, actual_returns)[0, 1]

@error_handler
def calculate_jensen_alpha(returns, market_returns, risk_free_rate):
    beta = calculate_beta(returns, market_returns)
    expected_return = risk_free_rate + beta * (np.mean(market_returns) - risk_free_rate)
    alpha = np.mean(returns) - expected_return
    return alpha
@error_handler
def calculate_treynor_ratio(returns, market_returns, risk_free_rate):
    beta = calculate_beta(returns, market_returns)
    return (np.mean(returns) - risk_free_rate) / beta
    @error_handler
def calculate_active_share(portfolio_weights, benchmark_weights):
    return np.sum(np.abs(portfolio_weights - benchmark_weights)) / 2

@error_handler
def calculate_portfolio_skewness(returns):
    return skew(returns)

@error_handler
def calculate_portfolio_kurtosis(returns):
    return kurtosis(returns)
@error_handler
def calculate_downside_correlation(returns1, returns2, threshold=0):
    downside_returns1 = returns1[returns1 < threshold]
    downside_returns2 = returns2[returns2 < threshold]
    return np.corrcoef(downside_returns1, downside_returns2)[0, 1]

@error_handler
def calculate_drawdown_at_risk(returns, confidence_level=0.95):
    drawdowns = 1 - (1 + returns).cumprod() / np.maximum.accumulate((1 + returns).cumprod())
    return np.percentile(drawdowns, 100 * confidence_level)

@error_handler
def calculate_gain_to_pain_ratio(returns):
    gains = np.sum(returns[returns > 0])
    pains = np.abs(np.sum(returns[returns < 0]))
    return gains / pains

@error_handler
def calculate_pain_index(returns):
    drawdowns = 1 - (1 + returns).cumprod() / np.maximum.accumulate((1 + returns).cumprod())
    return np.mean(drawdowns)

@error_handler
def calculate_calmar_ratio(returns, period=36):
    total_return = (1 + returns[-period:]).prod() - 1
    max_drawdown = calculate_maximum_drawdown(returns[-period:])
    return total_return / abs(max_drawdown)

@error_handler
def calculate_burke_ratio(returns, period=36):
    excess_returns = returns[-period:] - np.mean(returns[-period:])
    drawdowns = np.maximum(0, np.maximum.accumulate(excess_returns) - excess_returns)
    return np.mean(excess_returns) / np.sqrt(np.sum(drawdowns**2))

@error_handler
def calculate_martin_ratio(returns, period=36):
    ulcer_index = calculate_ulcer_index(returns[-period:])
    return np.mean(returns[-period:]) / ulcer_index

@error_handler
def calculate_rachev_ratio(returns, alpha=0.05, beta=0.05):
    var_alpha = np.percentile(returns, alpha * 100)
    var_beta = np.percentile(returns, (1 - beta) * 100)
    cvar_alpha = np.mean(returns[returns <= var_alpha])
    cvar_beta = np.mean(returns[returns >= var_beta])
    return -cvar_beta / cvar_alpha

@error_handler
def calculate_omega_ratio(returns, threshold=0):
    returns_above_threshold = returns[returns > threshold] - threshold
    returns_below_threshold = threshold - returns[returns <= threshold]
    return np.sum(returns_above_threshold) / np.sum(returns_below_threshold)

@error_handler
def calculate_sortino_ratio(returns, risk_free_rate, target_return=0):
    excess_return = np.mean(returns) - risk_free_rate
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return excess_return / downside_deviation

@error_handler
def calculate_kappa_three_ratio(returns, risk_free_rate, target_return=0):
    excess_return = np.mean(returns) - risk_free_rate
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return excess_return / (downside_deviation**3)**(1/3)

@error_handler
def calculate_upside_potential_ratio(returns, target_return=0):
    upside_returns = returns[returns > target_return] - target_return
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return np.mean(upside_returns) / downside_deviation

@error_handler
def calculate_hurst_exponent(returns, lags=range(2, 100)):
    tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@error_handler
def calculate_information_ratio(returns, benchmark_returns):
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    return np.mean(active_returns) / tracking_error

@error_handler
def calculate_modigliani_ratio(returns, benchmark_returns, risk_free_rate):
    portfolio_sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns, risk_free_rate)
    benchmark_volatility = np.std(benchmark_returns)
    return (portfolio_sharpe - benchmark_sharpe) * benchmark_volatility + benchmark_returns.mean()

@error_handler
def calculate_treynor_black_ratio(returns, benchmark_returns, risk_free_rate):
    alpha = calculate_jensen_alpha(returns, benchmark_returns, risk_free_rate)
    beta = calculate_beta(returns, benchmark_returns)
    return alpha / beta

@error_handler
def calculate_risk_adjusted_return(returns, risk_measure='std'):
    if risk_measure == 'std':
        risk = np.std(returns)
    elif risk_measure == 'downside_deviation':
        risk = calculate_downside_deviation(returns)
    elif risk_measure == 'var':
        risk = calculate_value_at_risk(returns)
    elif risk_measure == 'cvar':
        risk = calculate_conditional_value_at_risk(returns)
    else:
        raise ValueError("Invalid risk measure")
    
    return np.mean(returns) / risk

@error_handler
def calculate_maximum_drawdown_duration(returns):
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    drawdown_duration = np.zeros_like(drawdown)
    duration = 0
    for i in range(len(drawdown)):
        if drawdown[i] == 0:
            duration = 0
        else:
            duration += 1
        drawdown_duration[i] = duration
    return np.max(drawdown_duration)

@error_handler
def calculate_portfolio_concentration(weights):
    return 1 / np.sum(weights**2)

@error_handler
def calculate_diversification_ratio(weights, covariance_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    weighted_volatilities = np.sum(weights * np.sqrt(np.diag(covariance_matrix)))
    return weighted_volatilities / portfolio_volatility

@error_handler
def calculate_portfolio_beta(weights, stock_betas):
    return np.sum(weights * stock_betas)

@error_handler
def calculate_portfolio_alpha(weights, stock_alphas):
    return np.sum(weights * stock_alphas)

@error_handler
def calculate_portfolio_r_squared(returns, benchmark_returns):
    model = sm.OLS(returns, sm.add_constant(benchmark_returns)).fit()
    return model.rsquared

@error_handler
def calculate_portfolio_treynor_ratio(returns, benchmark_returns, risk_free_rate):
    portfolio_beta = calculate_beta(returns, benchmark_returns)
    return (np.mean(returns) - risk_free_rate) / portfolio_beta

@error_handler
def calculate_portfolio_information_ratio(returns, benchmark_returns):
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    return np.mean(active_returns) / tracking_error

@error_handler
def calculate_portfolio_sortino_ratio(returns, risk_free_rate, target_return=0):
    excess_return = np.mean(returns) - risk_free_rate
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return excess_return / downside_deviation

@error_handler
def calculate_portfolio_calmar_ratio(returns, period=36):
    total_return = (1 + returns[-period:]).prod() - 1
    max_drawdown = calculate_maximum_drawdown(returns[-period:])
    return total_return / abs(max_drawdown)

@error_handler
def calculate_portfolio_omega_ratio(returns, threshold=0):
    return calculate_omega_ratio(returns, threshold)

@error_handler
def calculate_portfolio_tail_ratio(returns, percentile=5):
    return calculate_tail_ratio(returns, percentile)

@error_handler
def calculate_portfolio_value_at_risk(returns, confidence_level=0.95):
    return calculate_value_at_risk(returns, confidence_level)

@error_handler
def calculate_portfolio_conditional_value_at_risk(returns, confidence_level=0.95):
    return calculate_conditional_value_at_risk(returns, confidence_level)

@error_handler
def calculate_portfolio_skewness(returns):
    return skew(returns)

@error_handler
def calculate_portfolio_kurtosis(returns):
    return kurtosis(returns)
@error_handler
def calculate_portfolio_expected_shortfall(returns, confidence_level=0.95):
    var = calculate_value_at_risk(returns, confidence_level)
    return np.mean(returns[returns <= var])

@error_handler
def calculate_portfolio_downside_risk(returns, target_return=0):
    downside_returns = returns[returns < target_return]
    return np.sqrt(np.mean((downside_returns - target_return)**2))

@error_handler
def calculate_portfolio_upside_risk(returns, target_return=0):
    upside_returns = returns[returns > target_return]
    return np.sqrt(np.mean((upside_returns - target_return)**2))

@error_handler
def calculate_portfolio_capture_ratio(returns, benchmark_returns):
    up_market = benchmark_returns > 0
    down_market = benchmark_returns < 0
    up_capture = np.mean(returns[up_market]) / np.mean(benchmark_returns[up_market])
    down_capture = np.mean(returns[down_market]) / np.mean(benchmark_returns[down_market])
    return up_capture / down_capture

@error_handler
def calculate_portfolio_up_capture_ratio(returns, benchmark_returns):
    up_market = benchmark_returns > 0
    return np.mean(returns[up_market]) / np.mean(benchmark_returns[up_market])

@error_handler
def calculate_portfolio_down_capture_ratio(returns, benchmark_returns):
    down_market = benchmark_returns < 0
    return np.mean(returns[down_market]) / np.mean(benchmark_returns[down_market])

@error_handler
def calculate_portfolio_batting_average(returns, benchmark_returns):
    return np.mean(returns > benchmark_returns)

@error_handler
def calculate_portfolio_win_loss_ratio(returns):
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return np.mean(wins) / abs(np.mean(losses))

@error_handler
def calculate_portfolio_pain_ratio(returns, risk_free_rate):
    excess_return = np.mean(returns) - risk_free_rate
    pain_index = calculate_pain_index(returns)
    return excess_return / pain_index

@error_handler
def calculate_portfolio_martin_ratio(returns, risk_free_rate):
    excess_return = np.mean(returns) - risk_free_rate
    ulcer_index = calculate_ulcer_index(returns)
    return excess_return / ulcer_index

@error_handler
def calculate_portfolio_sterling_ratio(returns, risk_free_rate, period=36):
    excess_return = np.mean(returns) - risk_free_rate
    max_drawdown = calculate_maximum_drawdown(returns[-period:])
    return excess_return / max_drawdown

@error_handler
def calculate_portfolio_burke_ratio(returns, risk_free_rate, period=36):
    excess_return = np.mean(returns) - risk_free_rate
    drawdowns = calculate_drawdowns(returns[-period:])
    return excess_return / np.sqrt(np.sum(drawdowns**2))

@error_handler
def calculate_portfolio_kappa_three_ratio(returns, risk_free_rate, target_return=0):
    excess_return = np.mean(returns) - risk_free_rate
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return excess_return / (downside_deviation**3)**(1/3)

@error_handler
def calculate_portfolio_omega_ratio(returns, threshold=0):
    return calculate_omega_ratio(returns, threshold)

@error_handler
def calculate_portfolio_sortino_ratio(returns, risk_free_rate, target_return=0):
    excess_return = np.mean(returns) - risk_free_rate
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return excess_return / downside_deviation

@error_handler
def calculate_portfolio_upside_potential_ratio(returns, target_return=0):
    upside_returns = returns[returns > target_return] - target_return
    downside_deviation = calculate_downside_deviation(returns, target_return)
    return np.mean(upside_returns) / downside_deviation

@error_handler
def calculate_portfolio_hurst_exponent(returns):
    return calculate_hurst_exponent(returns)

@error_handler
def calculate_portfolio_information_ratio(returns, benchmark_returns):
    return calculate_information_ratio(returns, benchmark_returns)

@error_handler
def calculate_portfolio_modigliani_ratio(returns, benchmark_returns, risk_free_rate):
    return calculate_modigliani_ratio(returns, benchmark_returns, risk_free_rate)

@error_handler
def calculate_portfolio_treynor_black_ratio(returns, benchmark_returns, risk_free_rate):
    return calculate_treynor_black_ratio(returns, benchmark_returns, risk_free_rate)

@error_handler
def calculate_portfolio_risk_adjusted_return(returns, risk_measure='std'):
    return calculate_risk_adjusted_return(returns, risk_measure)

@error_handler
def calculate_portfolio_maximum_drawdown_duration(returns):
    return calculate_maximum_drawdown_duration(returns)

@error_handler
def calculate_portfolio_diversification_ratio(weights, covariance_matrix):
    return calculate_diversification_ratio(weights, covariance_matrix)

@error_handler
def optimize_portfolio_black_litterman(returns, market_cap_weights, views, tau=0.05):
    # Расчет ковариационной матрицы
    cov_matrix = returns.cov()
    
    # Расчет равновесной доходности
    risk_aversion = 2.5  # Предполагаемый коэффициент неприятия риска
    equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_cap_weights)
    
    # Подготовка матрицы взглядов
    n_assets = len(returns.columns)
    n_views = len(views)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    
    for i, view in enumerate(views):
        P[i, returns.columns.get_loc(view['asset'])] = view['weight']
        Q[i] = view['expected_return']
    
    # Расчет матрицы неопределенности взглядов
    omega = np.dot(np.dot(P, cov_matrix), P.T) * tau
    
    # Расчет апостериорных доходностей
    term1 = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
    term2 = np.dot(np.linalg.inv(tau * cov_matrix), equilibrium_returns) + np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
    posterior_returns = np.dot(term1, term2)
    
    # Оптимизация портфеля с использованием апостериорных доходностей
    def objective(weights):
        portfolio_return = np.sum(posterior_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility  # Максимизация коэффициента Шарпа
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, market_cap_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_risk_parity(returns):
    cov_matrix = returns.cov()
    n_assets = len(returns.columns)
    
    def risk_parity_objective(weights):
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        asset_contributions = weights * (np.dot(cov_matrix, weights)) / portfolio_risk
        return np.sum((asset_contributions - asset_contributions.mean())**2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(risk_parity_objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_mean_cvar(returns, target_return, confidence_level=0.95):
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_returns = np.sum(returns * weights, axis=1)
        cvar = calculate_conditional_value_at_risk(portfolio_returns, confidence_level)
        return cvar
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_robust(returns, uncertainty=0.1):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        worst_case_return = portfolio_return - uncertainty * portfolio_volatility
        return -worst_case_return
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_hierarchical_risk_parity(returns):
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()
    distances = np.sqrt(0.5 * (1 - corr_matrix))
    
    # Иерархическая кластеризация
    linkage = hierarchy.linkage(distances, 'single')
    sorted_index = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
    
    sorted_returns = returns.iloc[:, sorted_index]
    sorted_cov_matrix = cov_matrix.iloc[sorted_index, sorted_index]
    
    # Рекурсивное бисекция для получения весов
    def bisect(items):
        if len(items) == 1:
            return [1]
        left = items[:len(items)//2]
        right = items[len(items)//2:]
        left_weight = 1 / np.sqrt(np.sum(sorted_cov_matrix.loc[left, left].values))
        right_weight = 1 / np.sqrt(np.sum(sorted_cov_matrix.loc[right, right].values))
        left_weights = bisect(left)
        right_weights = bisect(right)
        return [w * left_weight / (left_weight + right_weight) for w in left_weights] + \
               [w * right_weight / (left_weight + right_weight) for w in right_weights]
    
    weights = bisect(sorted_returns.columns)
    
    # Возвращаем веса в исходном порядке активов
    return pd.Series(weights, index=sorted_returns.columns).reindex(returns.columns)

@error_handler
def optimize_portfolio_factor_risk_parity(returns, factor_returns):
    # Расчет факторных нагрузок
    factor_loadings = calculate_factor_exposures(returns, factor_returns)
    
    # Расчет ковариационной матрицы факторов
    factor_cov_matrix = factor_returns.cov()
    
    # Функция для расчета факторных рисков
    def calculate_factor_risks(weights):
        portfolio_factor_loadings = np.dot(factor_loadings.T, weights)
        return np.sqrt(np.dot(portfolio_factor_loadings, np.dot(factor_cov_matrix, portfolio_factor_loadings)))
    
    # Целевая функция для оптимизации
    def objective(weights):
        factor_risks = calculate_factor_risks(weights)
        return np.sum((factor_risks - factor_risks.mean())**2)
    
    n_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_maximum_diversification(returns):
    cov_matrix = returns.cov()
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        asset_volatilities = np.sqrt(np.diag(cov_matrix))
        diversification_ratio = np.dot(weights, asset_volatilities) / portfolio_volatility
        return -diversification_ratio
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_minimum_correlation(returns):
    corr_matrix = returns.corr()
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_correlation = np.dot(weights.T, np.dot(corr_matrix, weights))
        return portfolio_correlation
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_maximum_decorrelation(returns):
    corr_matrix = returns.corr()
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_correlation = np.dot(weights.T, np.dot(corr_matrix, weights))
        return portfolio_correlation
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(x * returns.mean()) - returns.mean().mean()}
    )
    bounds = tuple((
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data

class StockGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(StockGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class StockGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=8):
        super(StockGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class StockSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(StockSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

@error_handler
def create_stock_graph(returns, threshold=0.5):
    corr_matrix = returns.corr()
    edges = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                edges.append((i, j))
                edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

@error_handler
def prepare_gnn_data(returns, features):
    edge_index = create_stock_graph(returns)
    x = torch.tensor(features.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

@error_handler
def train_gnn(model, data, target, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

@error_handler
def optimize_portfolio_gnn(returns, features, target_return, risk_tolerance):
    data = prepare_gnn_data(returns, features)
    model = StockGNN(num_features=features.shape[1], hidden_channels=64, num_classes=1)
    target = torch.tensor(returns.mean().values, dtype=torch.float).unsqueeze(1)
    
    train_gnn(model, data, target)
    
    model.eval()
    with torch.no_grad():
        predicted_returns = model(data.x, data.edge_index).squeeze().numpy()
    
    def objective(weights):
        portfolio_return = np.sum(predicted_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        return -(portfolio_return - risk_tolerance * portfolio_risk)
    
    n_assets = len(returns.columns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(predicted_returns * x) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def optimize_portfolio_gnn_ensemble(returns, features, target_return, risk_tolerance):
    data = prepare_gnn_data(returns, features)
    models = [
        StockGNN(num_features=features.shape[1], hidden_channels=64, num_classes=1),
        StockGAT(num_features=features.shape[1], hidden_channels=32, num_classes=1),
        StockSAGE(num_features=features.shape[1], hidden_channels=64, num_classes=1)
    ]
    target = torch.tensor(returns.mean().values, dtype=torch.float).unsqueeze(1)
    
    for model in models:
        train_gnn(model, data, target)
    
    predicted_returns = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).squeeze().numpy()
            predicted_returns.append(pred)
    
    ensemble_returns = np.mean(predicted_returns, axis=0)
    
    def objective(weights):
        portfolio_return = np.sum(ensemble_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        return -(portfolio_return - risk_tolerance * portfolio_risk)
    
    n_assets = len(returns.columns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(ensemble_returns * x) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

@error_handler
def evaluate_gnn_portfolio(returns, features, test_returns, optimization_func):
    train_data = prepare_gnn_data(returns, features)
    test_data = prepare_gnn_data(test_returns, features.loc[test_returns.index])
    
    weights = optimization_func(returns, features, target_return=0.1, risk_tolerance=0.5)
    
    portfolio_returns = np.sum(test_returns * weights, axis=1)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    max_drawdown = calculate_maximum_drawdown(portfolio_returns)
    
    return {
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Cumulative Return': (1 + portfolio_returns).prod() - 1
    }

@error_handler
def compare_gnn_portfolio_strategies(returns, features, test_returns):
    strategies = {
        'GNN': optimize_portfolio_gnn,
        'GNN Ensemble': optimize_portfolio_gnn_ensemble,
        'Mean-Variance': optimize_portfolio_markowitz,
        'Risk Parity': optimize_portfolio_risk_parity
    }
    
    results = {}
    for name, strategy in strategies.items():
        if 'GNN' in name:
            results[name] = evaluate_gnn_portfolio(returns, features, test_returns, strategy)
        else:
            weights = strategy(returns)
            portfolio_returns = np.sum(test_returns * weights, axis=1)
            results[name] = {
                'Sharpe Ratio': calculate_sharpe_ratio(portfolio_returns),
                'Max Drawdown': calculate_maximum_drawdown(portfolio_returns),
                'Cumulative Return': (1 + portfolio_returns).prod() - 1
            }
    
    return pd.DataFrame(results).T

# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    returns = load_stock_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'], '2010-01-01', '2021-12-31')
    features = load_fundamental_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'])
    
    # Разделение на обучающую и тестовую выборки
    train_returns = returns.loc[:'2020-12-31']
    test_returns = returns.loc['2021-01-01':]
    
    # Сравнение стратегий
    comparison = compare_gnn_portfolio_strategies(train_returns, features, test_returns)
    print(comparison)
@error_handler
def optimize_portfolio_gnn_multitask(returns, features, target_return, risk_tolerance):
    data = prepare_gnn_data(returns, features)
    
    class MultitaskGNN(torch.nn.Module):
        def __init__(self, num_features, hidden_channels, num_tasks):
            super(MultitaskGNN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.out = torch.nn.Linear(hidden_channels, num_tasks)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.out(x)

    model = MultitaskGNN(num_features=features.shape[1], hidden_channels=64, num_tasks=2)
    target_return = torch.tensor(returns.mean().values, dtype=torch.float)
    target_risk = torch.tensor(returns.std().values, dtype=torch.float)
    target = torch.stack([target_return, target_risk], dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).numpy()
        predicted_returns, predicted_risks = predictions[:, 0], predictions[:, 1]

    def objective(weights):
        portfolio_return = np.sum(predicted_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.diag(predicted_risks**2), weights)))
        return -(portfolio_return - risk_tolerance * portfolio_risk)

    n_assets = len(returns.columns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(predicted_returns * x) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

@error_handler
def optimize_portfolio_gnn_reinforcement(returns, features, num_episodes=1000, learning_rate=0.001):
    data = prepare_gnn_data(returns, features)
    
    class GNNPolicy(torch.nn.Module):
        def __init__(self, num_features, hidden_channels, num_assets):
            super(GNNPolicy, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.out = torch.nn.Linear(hidden_channels, num_assets)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return F.softmax(self.out(x), dim=1)

    num_assets = returns.shape[1]
    policy = GNNPolicy(num_features=features.shape[1], hidden_channels=64, num_assets=num_assets)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        policy.train()
        weights = policy(data.x, data.edge_index).detach().numpy()
        portfolio_return = np.sum(returns.mean() * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        reward = portfolio_return - 0.5 * portfolio_risk  # Пример функции вознаграждения

        optimizer.zero_grad()
        loss = -torch.sum(torch.log(policy(data.x, data.edge_index)) * torch.tensor(weights, dtype=torch.float32)) * reward
        loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {reward:.4f}")

    policy.eval()
    with torch.no_grad():
        final_weights = policy(data.x, data.edge_index).numpy()

    return final_weights

@error_handler
def optimize_portfolio_gnn_temporal(returns, features, sequence_length=30):
    from torch_geometric_temporal import TemporalData
    from torch_geometric_temporal.nn.recurrent import GConvGRU

    # Подготовка временных рядов данных
    edge_index = create_stock_graph(returns)
    node_features = []
    y = []
    for i in range(len(returns) - sequence_length):
        node_features.append(features.iloc[i:i+sequence_length].values)
        y.append(returns.iloc[i+sequence_length].values)
    
    node_features = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.tensor(np.array(y), dtype=torch.float)

    dataset = TemporalData(
        edge_index=edge_index,
        edge_attr=None,
        x=node_features,
        y=y
    )

    class TemporalGNN(torch.nn.Module):
        def __init__(self, node_features, hidden_channels, num_assets):
            super(TemporalGNN, self).__init__()
            self.recurrent = GConvGRU(node_features, hidden_channels, 2)
            self.linear = torch.nn.Linear(hidden_channels, num_assets)

        def forward(self, x, edge_index):
            h = self.recurrent(x, edge_index)
            h = F.relu(h)
            h = self.linear(h)
            return h

    model = TemporalGNN(node_features=features.shape[1], hidden_channels=64, num_assets=returns.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(200):
        cost = 0
        for time, snapshot in enumerate(dataset):
            y_hat = model(snapshot.x, snapshot.edge_index)
            cost = cost + criterion(y_hat, snapshot.y)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        predicted_returns = model(dataset[-1].x, dataset[-1].edge_index).numpy()

    def objective(weights):
        portfolio_return = np.sum(predicted_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.iloc[-sequence_length:].cov(), weights)))
        return -(portfolio_return - 0.5 * portfolio_risk)

    n_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(objective, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

@error_handler
def evaluate_portfolio_performance(weights, returns, risk_free_rate=0.02):
    portfolio_returns = np.sum(returns * weights, axis=1)
    excess_returns = portfolio_returns - risk_free_rate / 252  # Предполагаем дневные доходности

    performance = {
        'Total Return': (1 + portfolio_returns).prod() - 1,
        'Annualized Return': (1 + portfolio_returns).prod() ** (252 / len(returns)) - 1,
        'Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(excess_returns),
        'Sortino Ratio': calculate_sortino_ratio(excess_returns),
        'Max Drawdown': calculate_maximum_drawdown(portfolio_returns),
        'Calmar Ratio': calculate_calmar_ratio(portfolio_returns, risk_free_rate),
        'Omega Ratio': calculate_omega_ratio(portfolio_returns),
        'Tail Ratio': calculate_tail_ratio(portfolio_returns),
        'Value at Risk (95%)': calculate_value_at_risk(portfolio_returns),
        'Expected Shortfall (95%)': calculate_conditional_value_at_risk(portfolio_returns),
    }

    return performance

@error_handler
def run_portfolio_optimization_experiment(returns, features, test_returns, strategies):
    results = {}
    for name, strategy in strategies.items():
        print(f"Running strategy: {name}")
        if 'GNN' in name:
            weights = strategy(returns, features, target_return=0.1, risk_tolerance=0.5)
        else:
            weights = strategy(returns)
        
        train_performance = evaluate_portfolio_performance(weights, returns)
        test_performance = evaluate_portfolio_performance(weights, test_returns)
        
        results[name] = {
            'Train Performance': train_performance,
            'Test Performance': test_performance,
            'Weights': weights
        }
    
    return results

# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    returns = load_stock_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'], '2010-01-01', '2021-12-31')
    features = load_fundamental_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'])
    
    # Разделение на обучающую и тестовую выборки
    train_returns = returns.loc[:'2020-12-31']
    test_returns = returns.loc['2021-01-01':]
    
    strategies = {
        'GNN': optimize_portfolio_gnn,
        'GNN Ensemble': optimize_portfolio_gnn_ensemble,
        'GNN Multitask': optimize_portfolio_gnn_multitask,
        'GNN Reinforcement': optimize_portfolio_gnn_reinforcement,
        'GNN Temporal': optimize_portfolio_gnn_temporal,
        'Mean-Variance': optimize_portfolio_markowitz,
        'Risk Parity': optimize_portfolio_risk_parity,
        'Minimum Variance': optimize_portfolio_minimum_variance,
        'Maximum Sharpe': optimize_portfolio_maximum_sharpe,
        'Black-Litterman': optimize_portfolio_black_litterman,
    }
    
    results = run_portfolio_optimization_experiment(train_returns, features, test_returns, strategies)
    
    # Визуализация результатов
    for name, result in results.items():
        print(f"\nStrategy: {name}")
        print("Train Performance:")
        print(pd.DataFrame(result['Train Performance'], index=[0]))
        print("\nTest Performance:")
        print(pd.DataFrame(result['Test Performance'], index=[0]))
        print("\nPortfolio Weights:")
        print(pd.Series(result['Weights'], index=returns.columns))
        print("\n" + "="*50)
    
    # Сравнительный анализ стратегий
    train_sharpe_ratios = [result['Train Performance']['Sharpe Ratio'] for result in results.values()]
    test_sharpe_ratios = [result['Test Performance']['Sharpe Ratio'] for result in results.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(strategies)), train_sharpe_ratios, alpha=0.5, label='Train')
    plt.bar(range(len(strategies)), test_sharpe_ratios, alpha=0.5, label='Test')
    plt.xticks(range(len(strategies)), strategies.keys(), rotation=45, ha='right')
    plt.ylabel('Sharpe Ratio')
    plt.title('Comparison of Portfolio Optimization Strategies')
    plt.legend()
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()
