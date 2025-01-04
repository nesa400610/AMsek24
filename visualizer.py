import matplotlib.pyplot as plt
import seaborn as sns

def visualize_portfolio_performance(data, weights, initial_capital=10000):
    """
    Visualize portfolio performance.
    """
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

def visualize_correlation_matrix(data):
    """
    Visualize correlation matrix of stock returns.
    """
    returns = data.pct_change()
    corr_matrix = returns.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Stock Returns')
    plt.savefig('correlation_matrix.png')
    plt.close()
