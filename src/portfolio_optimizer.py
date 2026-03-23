import numpy as np

def min_variance_portfolio(cov_matrix):
    """
    Compute the minimum variance portfolio weights (no short sales constraints).
    Weights are positive and sum to 1.
    """
    p = cov_matrix.shape[0]
    # Solve for minimum variance: minimize w^T Σ w s.t. w^T 1 = 1, w >= 0
    # Using quadratic programming via scipy.optimize
    from scipy.optimize import minimize
    
    def objective(w):
        return w @ cov_matrix @ w
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(p)]
    w0 = np.ones(p) / p
    
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        # Fallback to equal weights
        return np.ones(p) / p
