"""
Finance PDF Report Generator
"""

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from .market_data_loaders import load_market_data_from_yahoo
from .technical_indicators_calculator import compute_indicators
from .price_chart_plotter import plot_price_chart



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