import numpy as np
from scipy.optimize import minimize

def optimize_portfolio_markowitz(returns, target_return, risk_tolerance):
    """
    Optimize portfolio using Markowitz model.
    """
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

def optimize_portfolio_gnn(model, data, risk_tolerance):
    """
    Optimize portfolio using GNN predictions.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
    weights = torch.softmax(predictions, dim=0).numpy()
    return weights
