import pandas as pd
import yfinance as yf

def download_data(ticker, start_date, end_date, interval):
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    data = data.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1)
    data.index = data.index.tz_localize(None).normalize()
    return data
