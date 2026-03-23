import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_asset_data(symbols, start_date, end_date=None):
    """
    Fetch daily adjusted close prices from Yahoo Finance.
    Returns DataFrame with dates as index and symbols as columns.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    
    if len(symbols) == 1:
        prices = data['Adj Close'].to_frame(name=symbols[0])
    else:
        # Handles newer yfinance multi-index structures safely
        prices = pd.DataFrame({sym: data[sym]['Adj Close'] for sym in symbols if sym in data})
    
    prices = prices.dropna()
    return prices

def load_from_csv(filepath):
    """Load pre-saved CSV with Date index."""
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    return df

def compute_returns(prices):
    """Compute daily log returns."""
    returns = np.log(prices).diff().dropna()
    return returns
