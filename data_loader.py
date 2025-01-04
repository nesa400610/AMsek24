import yfinance as yf
import pandas as pd

def load_stock_data(tickers, start_date, end_date):
    """
    Load stock data for given tickers and date range.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def load_fundamental_data(tickers):
    """
    Load fundamental data for given tickers.
    """
    fundamental_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        fundamental_data[ticker] = stock.info
    return pd.DataFrame(fundamental_data).T
