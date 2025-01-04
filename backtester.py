import numpy as np
import pandas as pd

def backtest_strategy(data, weights, initial_capital=10000):
    """
    Backtest a portfolio strategy.
    """
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
