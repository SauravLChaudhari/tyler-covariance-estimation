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
        prices = data['Adj Close']
    
    prices = prices.dropna()
    return prices

def compute_returns(prices):
    """Compute daily log returns."""
    returns = np.log(prices).diff().dropna()
    return returns
