#!/usr/bin/env python3
"""
Finance Report CLI
A simple command-line interface for generating financial analysis reports
"""

import os
import sys
import subprocess
import platform
from .pdf_report.finance_pdf_report import generate_finance_report


def open_pdf(pdf_path):
    """
    Open PDF file in the default system viewer.
    
    Args:
        pdf_path (str): Path to the PDF file
    """
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["open", pdf_path], check=True)
        elif system == "Windows":
            os.startfile(pdf_path)
        elif system == "Linux":
            subprocess.run(["xdg-open", pdf_path], check=True)
        else:
            print(f"Cannot automatically open PDF on {system}. Please open {pdf_path} manually.")
    except Exception as e:
        print(f"Could not open PDF automatically: {e}")
        print(f"Please open the file manually: {pdf_path}")


def get_tickers():
    """
    Collect up to 5 ticker symbols from user input.
    
    Returns:
        list: List of valid ticker symbols
    """
    print("=" * 60)
    print("📈 FINANCIAL ANALYSIS REPORT GENERATOR 📈")
    print("=" * 60)
    print()
    print("This tool will generate a comprehensive financial analysis report")
    print("including technical indicators, charts, and price analysis.")
    print()
    print("Please enter up to 5 stock ticker symbols (e.g., AAPL, MSFT, GOOGL)")
    print("Press Enter after each ticker, or type 'done' when finished.")
    print("Type 'quit' to exit without generating a report.")
    print()
    
    tickers = []
    
    for i in range(5):
        while True:
            if i == 0:
                prompt = f"Enter ticker #{i+1} (required): "
            else:
                prompt = f"Enter ticker #{i+1} (optional, or 'done' to finish): "
            
            ticker = input(prompt).strip().upper()
            
            # Handle exit conditions
            if ticker.lower() == 'quit':
                print("Exiting without generating report.")
                return None
            
            if ticker.lower() == 'done':
                if i == 0:
                    print("At least one ticker is required. Please enter a ticker symbol.")
                    continue
                else:
                    break
            
            # Validate ticker input
            if not ticker:
                if i == 0:
                    print("At least one ticker is required. Please enter a ticker symbol.")
                    continue
                else:
                    break
            
            # Basic validation for ticker format
            if not ticker.isalpha() or len(ticker) > 10:
                print("Invalid ticker format. Please enter a valid stock symbol (letters only, max 10 characters).")
                continue
            
            # Check for duplicates
            if ticker in tickers:
                print(f"'{ticker}' has already been entered. Please enter a different ticker.")
                continue
            
            tickers.append(ticker)
            print(f"✓ Added {ticker}")
            break
    
    return tickers


def main():
    """
    Main CLI function that orchestrates the financial report generation process.
    """
    try:
        # Get ticker symbols from user
        tickers = get_tickers()
        
        if not tickers:
            return
        
        print()
        print("=" * 60)
        print(f"Generating financial report for: {', '.join(tickers)}")
        print("=" * 60)
        print()
        print("📊 Loading market data and computing technical indicators...")
        print("This may take a few moments depending on your internet connection.")
        print()
        
        # Generate the financial report
        pdf_filename = generate_finance_report(tickers)
        
        print()
        print("=" * 60)
        print("🎉 REPORT GENERATION COMPLETE! 🎉")
        print("=" * 60)
        print(f"📄 Report saved as: {pdf_filename}")
        print()
        
        # Ask user if they want to open the PDF
        while True:
            open_choice = input("Would you like to open the PDF report now? (y/n): ").strip().lower()
            if open_choice in ['y', 'yes']:
                print("Opening PDF report...")
                open_pdf(pdf_filename)
                break
            elif open_choice in ['n', 'no']:
                print(f"Report saved successfully. You can open {pdf_filename} manually.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        print()
        print("Thank you for using the Financial Analysis Report Generator!")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()