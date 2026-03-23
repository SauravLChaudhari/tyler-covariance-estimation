import matplotlib.pyplot as plt
import numpy as np

def plot_portfolio_weights(weights_sample, weights_robust, asset_names):
    """
    Bar plot comparing weights from sample covariance and robust covariance.
    """
    x = np.arange(len(asset_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, weights_sample, width, label='Sample Covariance', alpha=0.8)
    ax.bar(x + width/2, weights_robust, width, label='Tyler (Robust)', alpha=0.8)
    
    ax.set_ylabel('Portfolio Weight')
    ax.set_title('Minimum Variance Portfolio Weights')
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(returns_sample, returns_robust, benchmark=None):
    """
    Plot cumulative returns of portfolios.
    """
    cum_sample = (1 + returns_sample).cumprod()
    cum_robust = (1 + returns_robust).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_sample.index, cum_sample, label='Sample Covariance', linewidth=2)
    plt.plot(cum_robust.index, cum_robust, label='Tyler (Robust)', linewidth=2)
    if benchmark is not None:
        cum_bench = (1 + benchmark).cumprod()
        plt.plot(cum_bench.index, cum_bench, label='Benchmark (SPY)', linestyle='--', alpha=0.7)
    
    plt.ylabel('Cumulative Return')
    plt.title('Portfolio Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
