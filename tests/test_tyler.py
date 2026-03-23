import numpy as np
from src.tyler_estimator import tyler_estimator

def test_tyler_on_gaussian():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    cov_tyler = tyler_estimator(X, max_iter=20, normalize='trace')
    # Should be close to identity (since true covariance is identity)
    assert np.allclose(cov_tyler, np.eye(5), atol=0.2)

def test_tyler_with_outliers():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    # Add extreme outliers (10% of samples)
    n_out = 20
    X[:n_out] += 10 * np.random.randn(n_out, 5)
    cov_tyler = tyler_estimator(X, max_iter=30)
    cov_sample = np.cov(X, rowvar=False)
    # Sample covariance should be inflated; Tyler should be less affected
    # Check trace: sample trace should be larger than Tyler's
    assert np.trace(cov_sample) > np.trace(cov_tyler)
