"""
Finance PDF Report Generator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


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


def plot_price_chart(ticker, df, figsize=(12, 8)):
    """
    Generates a comprehensive stock price chart for a given ticker that includes 
    the close price, 20-day and 50-day SMAs, and the Bollinger Bands.
    
    Args:
        ticker (str): Ticker symbol for chart title
        df (pd.DataFrame): DataFrame with indicators already computed from compute_indicators()
        figsize (tuple): Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object that can be saved or processed
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine x-axis data (use Date column if available, otherwise index)
    if 'Date' in df.columns:
        x_data = pd.to_datetime(df['Date'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        x_data = df.index
        ax.set_xlabel('Time Period')
    
    # Plot close price
    ax.plot(x_data, df['Close'], 
            label='Close Price', color='black', linewidth=1.5)
    
    # Plot SMAs (only where they exist)
    sma_20_mask = ~pd.isna(df['SMA_20'])
    if sma_20_mask.any():
        ax.plot(x_data[sma_20_mask], df.loc[sma_20_mask, 'SMA_20'], 
                label='20-day SMA', color='blue', linewidth=1, alpha=0.8)
    
    sma_50_mask = ~pd.isna(df['SMA_50'])
    if sma_50_mask.any():
        ax.plot(x_data[sma_50_mask], df.loc[sma_50_mask, 'SMA_50'], 
                label='50-day SMA', color='red', linewidth=1, alpha=0.8)
    
    # Plot Bollinger Bands (only where they exist)
    bb_mask = ~pd.isna(df['BB_Upper'])
    if bb_mask.any():
        ax.plot(x_data[bb_mask], df.loc[bb_mask, 'BB_Upper'], 
                label='Bollinger Upper', color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.plot(x_data[bb_mask], df.loc[bb_mask, 'BB_Lower'], 
                label='Bollinger Lower', color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
        
        # Fill area between Bollinger Bands
        ax.fill_between(x_data[bb_mask], 
                       df.loc[bb_mask, 'BB_Lower'], 
                       df.loc[bb_mask, 'BB_Upper'], 
                       alpha=0.1, color='gray', label='Bollinger Bands')
    
    # Customize chart
    ax.set_title(f'{ticker} - Stock Price Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price ($)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    plt.tight_layout()
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    
    return fig


def generate_finance_report():
    """Generate a financial PDF report."""
    pass


if __name__ == "__main__":
    generate_finance_report()