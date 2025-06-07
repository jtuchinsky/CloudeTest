"""
Finance PDF Report Generator
"""

import pandas as pd
import numpy as np


def compute_indicators(df):
    """
    Computes technical indicators for the DataFrame:
    - 20-day and 50-day Simple Moving Averages (SMA)
    - Bollinger Bands (using 20-day SMA ±2 standard deviations)
    - MACD (12-day EMA minus 26-day EMA) and its 9-day Signal Line
    - RSI (Relative Strength Index) over a 14-day lookback period
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column containing price data
        
    Returns:
        pd.DataFrame: Original DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20']
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df


def generate_finance_report():
    """Generate a financial PDF report."""
    pass


if __name__ == "__main__":
    generate_finance_report()