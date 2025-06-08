import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages
from claudetest.pdf_report.finance_pdf_report import generate_finance_report, create_cover_page
from claudetest.pdf_report.market_data_loaders import load_market_data_from_yahoo





class TestGenerateFinanceReport:
    """Test suite for basic generate_finance_report validation."""
    
    def test_generate_finance_report_empty_ticker_list(self):
        """Test report generation with empty ticker list."""
        with pytest.raises(ValueError) as exc_info:
            generate_finance_report([])
        
        assert "tickers must be a non-empty list" in str(exc_info.value)
    
    def test_generate_finance_report_invalid_input_type(self):
        """Test report generation with invalid input types."""
        # Test with string instead of list
        with pytest.raises(ValueError) as exc_info:
            generate_finance_report("AAPL")
        
        assert "tickers must be a non-empty list" in str(exc_info.value)
        
        # Test with None
        with pytest.raises(ValueError) as exc_info:
            generate_finance_report(None)
        
        assert "tickers must be a non-empty list" in str(exc_info.value)
    
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_all_tickers_fail(self, mock_print, mock_load_data):
        """Test report generation when all tickers fail."""
        # Setup mock to always raise ValueError
        mock_load_data.side_effect = ValueError("No data available")
        
        # Test function should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            generate_finance_report(['INVALID1', 'INVALID2'])
        
        assert "Failed to process any tickers" in str(exc_info.value)
    


class TestCreateCoverPage:
    """Test suite for the create_cover_page function."""
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_cover_page_basic_functionality(self, mock_close, mock_savefig):
        """Test that create_cover_page creates and saves a figure."""
        # Create a mock PDF object
        mock_pdf = MagicMock()
        
        # Test function
        create_cover_page(mock_pdf)
        
        # Verify that savefig was called on the PDF object
        assert mock_pdf.savefig.called
        
        # Verify that close was called
        assert mock_close.called
    
    @patch('matplotlib.pyplot.text')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.figure')
    def test_create_cover_page_content(self, mock_figure, mock_axis, mock_text):
        """Test that create_cover_page creates proper content."""
        # Setup mock figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Create a mock PDF object
        mock_pdf = MagicMock()
        
        # Test function
        create_cover_page(mock_pdf)
        
        # Verify figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(11.69, 8.27))
        
        # Verify axis was turned off
        mock_axis.assert_called_once_with('off')
        
        # Verify text elements were added
        assert mock_text.call_count >= 4  # Should have at least 4 text elements
        
        # Check for expected text content in calls
        text_calls = [call.args for call in mock_text.call_args_list]
        
        # Should contain main title
        title_found = any("Financial Analysis Report" in str(call) for call in text_calls)
        assert title_found, "Should contain main title"


class TestGenerateFinanceReportPDF:
    """Updated test suite for the PDF-generating generate_finance_report function."""
    
    @pytest.fixture
    def mock_yahoo_data(self):
        """Create mock Yahoo Finance data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = np.random.uniform(90, 110, len(dates))
        
        mock_hist = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        mock_hist.index.name = 'Date'
        
        return mock_hist
    
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
    @patch('builtins.print')
    def test_generate_finance_report_pdf_creation(self, mock_print, mock_ticker, mock_pdf_pages, mock_yahoo_data):
        """Test that generate_finance_report creates a PDF file."""
        # Setup mocks
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_yahoo_data
        mock_ticker.return_value = mock_ticker_instance
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        # Test function
        result = generate_finance_report(['AAPL'])
        
        # Verify PDF creation
        mock_pdf_pages.assert_called_once_with("financial_report.pdf")
        
        # Verify result is PDF filename
        assert result == "financial_report.pdf"
    
    @patch('claudetest.pdf_report.finance_pdf_report.create_cover_page')
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_cover_page_called(self, mock_print, mock_load_data, mock_pdf_pages, mock_create_cover, mock_yahoo_data):
        """Test that cover page is created."""
        # Setup mocks
        mock_load_data.return_value = mock_yahoo_data
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        # Test function
        generate_finance_report(['AAPL'])
        
        # Verify cover page was created
        mock_create_cover.assert_called_once_with(mock_pdf_context)
    
    @patch('claudetest.pdf_report.finance_pdf_report.plot_price_chart')
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_charts_added_to_pdf(self, mock_print, mock_load_data, mock_pdf_pages, mock_plot_chart, mock_yahoo_data):
        """Test that charts are created and added to PDF."""
        # Setup mocks
        mock_load_data.return_value = mock_yahoo_data
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        mock_figure = MagicMock()
        mock_plot_chart.return_value = mock_figure
        
        # Test function
        generate_finance_report(['AAPL', 'MSFT'])
        
        # Verify charts were created for each ticker
        assert mock_plot_chart.call_count == 2
        
        # Verify charts were saved to PDF (plus 1 for cover page = 3 total)
        assert mock_pdf_context.savefig.call_count == 3
    
    @patch('matplotlib.pyplot.close')
    @patch('claudetest.pdf_report.finance_pdf_report.plot_price_chart')
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_figures_cleaned_up(self, mock_print, mock_load_data, mock_pdf_pages, mock_plot_chart, mock_close, mock_yahoo_data):
        """Test that matplotlib figures are properly closed."""
        # Setup mocks
        mock_load_data.return_value = mock_yahoo_data
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        mock_figure = MagicMock()
        mock_plot_chart.return_value = mock_figure
        
        # Test function
        generate_finance_report(['AAPL'])
        
        # Verify figure was closed
        mock_close.assert_called_with(mock_figure)
    
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_mixed_success_failure_pdf(self, mock_print, mock_load_data, mock_pdf_pages, mock_yahoo_data):
        """Test PDF generation with some failed tickers."""
        # Setup mock to succeed for AAPL but fail for INVALID
        def mock_load_data_side_effect(ticker):
            if ticker == 'AAPL':
                return mock_yahoo_data
            else:
                raise ValueError(f"No data available for ticker '{ticker}'")
        
        mock_load_data.side_effect = mock_load_data_side_effect
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        # Test function
        result = generate_finance_report(['AAPL', 'INVALID'])
        
        # Verify PDF was still created for successful ticker
        assert result == "financial_report.pdf"
        
        # Verify PDF creation was attempted
        mock_pdf_pages.assert_called_once_with("financial_report.pdf")
    
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_all_tickers_fail_no_pdf(self, mock_print, mock_load_data, mock_pdf_pages):
        """Test that no PDF is created when all tickers fail."""
        # Setup mock to always raise ValueError
        mock_load_data.side_effect = ValueError("No data available")
        
        # Test function should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            generate_finance_report(['INVALID1', 'INVALID2'])
        
        assert "Failed to process any tickers" in str(exc_info.value)
        
        # Verify PDF creation was not attempted
        mock_pdf_pages.assert_not_called()
    
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('builtins.print')
    def test_generate_finance_report_empty_ticker_list_no_pdf(self, mock_print, mock_pdf_pages):
        """Test that no PDF is created with empty ticker list."""
        with pytest.raises(ValueError) as exc_info:
            generate_finance_report([])
        
        assert "tickers must be a non-empty list" in str(exc_info.value)
        
        # Verify PDF creation was not attempted
        mock_pdf_pages.assert_not_called()
    
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_return_value(self, mock_print, mock_load_data, mock_pdf_pages, mock_yahoo_data):
        """Test that function returns correct PDF filename."""
        # Setup mocks
        mock_load_data.return_value = mock_yahoo_data
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        # Test function
        result = generate_finance_report(['AAPL'])
        
        # Verify correct return value
        assert result == "financial_report.pdf"
        assert isinstance(result, str)
    
    @patch('claudetest.pdf_report.finance_pdf_report.PdfPages')
    @patch('claudetest.pdf_report.market_data_loaders.load_market_data_from_yahoo')
    @patch('builtins.print')
    def test_generate_finance_report_progress_messages(self, mock_print, mock_load_data, mock_pdf_pages, mock_yahoo_data):
        """Test that appropriate progress messages are printed."""
        # Setup mocks
        mock_load_data.return_value = mock_yahoo_data
        
        mock_pdf_context = MagicMock()
        mock_pdf_pages.return_value.__enter__.return_value = mock_pdf_context
        
        # Test function
        generate_finance_report(['AAPL'])
        
        # Verify progress messages were printed
        assert mock_print.called
        
        # Check for expected messages
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        
        # Should have data loading message
        loading_messages = [msg for msg in print_calls if 'Loading market data' in msg]
        assert len(loading_messages) >= 1
        
        # Should have PDF generation message
        pdf_messages = [msg for msg in print_calls if 'Generating PDF report' in msg]
        assert len(pdf_messages) >= 1
        
        # Should have chart addition message
        chart_messages = [msg for msg in print_calls if 'Adding chart' in msg]
        assert len(chart_messages) >= 1