import os
import numpy as np
import pandas as pd

def generate_synthetic_market_data(output_path='../data/synthetic_market_data.csv'):
    """
    Generates synthetic daily prices for a basket of stocks.
    Injects a period of extreme volatility ('crash') to demonstrate
    the robustness of Tyler's M-estimator compared to standard covariance.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', end='2020-12-31', freq='B')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'WMT', 'SPY']
    
    # 1. Base daily returns (~15% annualized volatility)
    daily_vol = 0.15 / np.sqrt(252)
    returns = np.random.normal(0.0005, daily_vol, (len(dates), len(symbols)))
    
    # 2. Inject a "Flash Crash" / Heavy Outliers (March 2020)
    # The standard sample covariance will be heavily distorted by this.
    # Tyler's estimator will naturally down-weight these extreme observations.
    crash_idx = np.where((dates.year == 2020) & (dates.month == 3))[0]
    returns[crash_idx] += np.random.normal(-0.02, 0.08, (len(crash_idx), len(symbols)))
    
    # 3. Convert returns to simulated prices (starting at 100)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    df = pd.DataFrame(prices, index=dates, columns=symbols)
    df.index.name = 'Date'
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    
    print(f"Generated {len(dates)} days of synthetic data at {output_path}")
    print("Injected heavy-tailed outliers in March 2020 to test robust estimation.")

if __name__ == '__main__':
    generate_synthetic_market_data()
