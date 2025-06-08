"""
Finance PDF Report Generator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from .market_data_loaders import load_market_data_from_yahoo
from .technical_indicators_calculator import compute_indicators


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

def generate_finance_report(tickers):
    """
    Generate a comprehensive financial PDF report for multiple tickers.
    For each ticker, loads market data, computes indicators, and creates price charts.
    Saves everything to a PDF file.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        
    Returns:
        str: Path to the generated PDF file
        
    Raises:
        ValueError: If tickers list is empty or contains invalid tickers
        Exception: If there's an error processing any ticker
    """
    if not tickers or not isinstance(tickers, list):
        raise ValueError("tickers must be a non-empty list of ticker symbols")
    
    # Assign PDF filename
    pdf_filename = "financial_report.pdf"
    
    results = {}
    failed_tickers = []
    
    # Load market data for each ticker
    for ticker in tickers:
        try:
            print(f"Loading market data for {ticker}...")
            
            # Load market data
            market_data = load_market_data_from_yahoo(ticker)
            
            # Compute technical indicators
            indicators_data = compute_indicators(market_data)
            
            # Store result (we'll create charts when saving to PDF)
            results[ticker] = indicators_data
            
            print(f"✓ Successfully loaded data for {ticker}")
            
        except ValueError as e:
            print(f"✗ Failed to load data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            continue
            
        except Exception as e:
            print(f"✗ Error loading data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            continue
    
    # Check if any tickers were successfully processed
    if not results:
        raise ValueError(f"Failed to process any tickers. Failed tickers: {failed_tickers}")
    
    # Create PDF report
    print(f"\nGenerating PDF report: {pdf_filename}")
    
    with PdfPages(pdf_filename) as pdf:
        # Create cover page
        create_cover_page(pdf)
        
        # Create charts for each ticker and add to PDF
        for ticker, data in results.items():
            print(f"Adding chart for {ticker} to PDF...")
            chart_figure = plot_price_chart(ticker, data)
            pdf.savefig(chart_figure)
            plt.close(chart_figure)  # Clean up figure
    
    # Report summary
    successful_count = len(results)
    total_count = len(tickers)
    print(f"\nReport Summary:")
    print(f"Successfully processed: {successful_count}/{total_count} tickers")
    print(f"PDF report saved as: {pdf_filename}")
    
    if failed_tickers:
        print(f"Failed tickers: {failed_tickers}")
    
    return pdf_filename

def create_cover_page(pdf):
    """
    Creates and saves a cover page into the PDF report.
    """
    fig = plt.figure(figsize=(11.69, 8.27))
    plt.axis('off')
    plt.text(0.5, 0.7, "Financial Analysis Report", fontsize=24, ha='center')
    plt.text(0.5, 0.62, "Analysis of 5 Stocks from Yahoo Finance", fontsize=16, ha='center')
    plt.text(0.5, 0.5, "Includes Technical Indicators: SMA, Bollinger Bands, MACD, RSI", fontsize=12, ha='center')
    plt.text(0.5, 0.4, "Generated with Python and matplotlib", fontsize=10, ha='center')
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    generate_finance_report()