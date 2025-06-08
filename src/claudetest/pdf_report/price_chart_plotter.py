"""
Price Chart Plotter
Functions for creating and customizing financial price charts with technical indicators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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