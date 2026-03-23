import numpy as np
from scipy.linalg import det, inv

def tyler_estimator(X, max_iter=100, tol=1e-6, normalize='trace'):
    """
    Compute Tyler's robust covariance estimate for data matrix X (n_samples x n_features).
    
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix (returns).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance (relative change in Frobenius norm).
    normalize : str
        'trace' or 'det': normalisation to ensure identifiability.
    
    Returns
    -------
    S : ndarray, shape (p, p)
        Robust scatter matrix.
    """
    n, p = X.shape
    # Initialise S with identity
    S = np.eye(p)
    
    for _ in range(max_iter):
        # Compute distances d_i = x_i^T S^{-1} x_i
        try:
            inv_S = inv(S)
        except np.linalg.LinAlgError:
            inv_S = np.linalg.pinv(S)
        
        dist = np.einsum('ij,ij->i', X @ inv_S, X)  # equivalent to diag(X @ inv_S @ X.T)
        # Avoid zero distances
        dist = np.maximum(dist, 1e-12)
        
        # Update S
        weights = 1.0 / dist
        S_new = (p / n) * (X.T @ (weights[:, None] * X))
        
        # Normalise
        if normalize == 'trace':
            S_new = S_new / np.trace(S_new) * p
        elif normalize == 'det':
            S_new = S_new / (det(S_new) ** (1.0/p))
        
        # Check convergence
        if np.linalg.norm(S_new - S, 'fro') < tol * np.linalg.norm(S, 'fro'):
            break
        S = S_new
    
    return S
