# Robust Covariance Estimation via Tyler's M‑Estimator

## Signal‑Processing to Quant Translation

In RF sensing, we use **robust M‑estimators** to estimate covariance matrices in the presence of interference (jammers) or outliers.  
Tyler's estimator is a particularly powerful method: it iteratively reweights observations to down‑weigh outliers, producing a scatter matrix that is robust to heavy‑tailed noise.

In finance, the sample covariance matrix is notoriously fragile: a single extreme return (e.g., during a flash crash) can distort the entire estimate, leading to poor portfolio decisions.  
By applying Tyler's estimator to asset returns, we obtain a **robust covariance matrix** that ignores outliers and yields more stable portfolios.

## Mathematical Model

Given `n` observations (returns) `x₁,…,xₙ` in `ℝᵖ`, Tyler's estimator finds a scatter matrix `S` satisfying:
`S = (p / n) * Σ_i [ (x_i x_iᵀ) / (x_iᵀ S⁻¹ x_i) ]`

This is solved by fixed‑point iteration:

1. Initialize `S` (e.g., identity matrix).
2. For each iteration, update:
   `S_new = (p / n) * Σ_i [ (x_i x_iᵀ) / (x_iᵀ S⁻¹ x_i) ]`
3. Normalize to have unit determinant (or trace) to avoid scaling ambiguity.

The solution is the **MLE** for a multivariate t‑distribution and is robust to outliers.

## Repository Contents

- `src/` – core modules
  - `tyler_estimator.py`: implementation of Tyler's iterative algorithm
  - `data_loader.py`: fetches asset data / loads local data
  - `portfolio_optimizer.py`: minimum variance portfolio with robust covariance
  - `visualization.py`: plots portfolio weights and performance
  - `generate_dummy_data.py`: simulates market data with a built-in "crash"
- `notebooks/` – interactive demo with backtest
- `tests/` – unit tests

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
