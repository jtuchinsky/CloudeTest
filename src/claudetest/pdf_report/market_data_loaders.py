"""
Market Data Loaders
Functions for loading market data from various sources.
"""

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


def load_market_data_from_yahoo(ticker):
    """
    Loads market data from Yahoo Finance for the given ticker.
    Returns DataFrame with daily close prices for the past year from today.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        pd.DataFrame: DataFrame with Date and Close columns for the past year

    Raises:
        ValueError: If ticker is invalid or no data is available
        Exception: If there's an error fetching data from Yahoo Finance
    """
    try:
        # Calculate date range (past year from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # Create ticker object and fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        # Check if data was returned
        if hist.empty:
            raise ValueError(f"No data available for ticker '{ticker}'. Please "
                             f"check if the ticker symbol is valid.")

        # Reset index to make Date a column and select only Close price
        df = hist.reset_index()[['Date', 'Close']]

        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by date (oldest first)
        df = df.sort_values('Date').reset_index(drop=True)

        return df

    except ValueError as e:
        # Re-raise ValueError as-is (from our own validation or yfinance)
        raise e
    except Exception as e:
        if "No data found" in str(e) or "Invalid ticker" in str(e):
            raise ValueError(f"Invalid ticker symbol '{ticker}' or no data available.")
        else:
            raise Exception(f"Error fetching data for ticker '{ticker}': {str(e)}")
