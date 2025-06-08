import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages
from claudetest.pdf_report.finance_pdf_report import compute_indicators, plot_price_chart, generate_finance_report, create_cover_page
from claudetest.pdf_report.market_data_loaders import load_market_data_from_yahoo


class TestComputeIndicators:
    """Test suite for the compute_indicators function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data with some trend and volatility
        base_price = 100
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data for edge case testing."""
        return pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
        })
    
    def test_compute_indicators_basic_functionality(self, sample_data):
        """Test that compute_indicators adds all expected columns."""
        result = compute_indicators(sample_data)
        
        expected_columns = [
            'Close', 'SMA_20', 'SMA_50', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Column {col} missing from result"
    
    def test_sma_calculation(self, minimal_data):
        """Test Simple Moving Average calculations."""
        result = compute_indicators(minimal_data)
        
        # SMA_20 should be NaN for first 19 rows (0-indexed)
        assert pd.isna(result['SMA_20'].iloc[0])
        
        # For minimal data (10 points), SMA_20 should be NaN due to insufficient data
        assert pd.isna(result['SMA_20'].iloc[-1])
        
        # Test with sufficient data for SMA calculation
        large_data = pd.DataFrame({'Close': list(range(100, 150))})  # 50 data points
        large_result = compute_indicators(large_data)
        
        # SMA_20 at position 19 should equal mean of first 20 values
        expected_sma_20 = large_data['Close'].iloc[:20].mean()
        assert abs(large_result['SMA_20'].iloc[19] - expected_sma_20) < 0.01
    
    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculations."""
        result = compute_indicators(sample_data)
        
        # BB_Middle should equal SMA_20
        pd.testing.assert_series_equal(
            result['BB_Middle'], 
            result['SMA_20'], 
            check_names=False
        )
        
        # Upper band should be greater than middle, lower should be less
        valid_rows = ~pd.isna(result['BB_Middle'])
        assert (result.loc[valid_rows, 'BB_Upper'] >= result.loc[valid_rows, 'BB_Middle']).all()
        assert (result.loc[valid_rows, 'BB_Lower'] <= result.loc[valid_rows, 'BB_Middle']).all()
    
    def test_macd_calculation(self, sample_data):
        """Test MACD calculations."""
        result = compute_indicators(sample_data)
        
        # MACD Histogram should equal MACD - MACD_Signal
        macd_diff = result['MACD'] - result['MACD_Signal']
        pd.testing.assert_series_equal(
            result['MACD_Histogram'], 
            macd_diff, 
            check_names=False
        )
        
        # MACD should have some non-null values
        assert not result['MACD'].isna().all()
    
    def test_rsi_bounds(self, sample_data):
        """Test that RSI values are within expected bounds (0-100)."""
        result = compute_indicators(sample_data)
        
        # Remove NaN values for testing
        rsi_values = result['RSI'].dropna()
        
        assert (rsi_values >= 0).all(), "RSI values should be >= 0"
        assert (rsi_values <= 100).all(), "RSI values should be <= 100"
    
    def test_original_data_preserved(self, sample_data):
        """Test that original DataFrame is not modified."""
        original_columns = sample_data.columns.tolist()
        original_values = sample_data.copy()
        
        result = compute_indicators(sample_data)
        
        # Original DataFrame should be unchanged
        pd.testing.assert_frame_equal(sample_data, original_values)
        
        # Result should contain original columns plus new ones
        for col in original_columns:
            assert col in result.columns
    
    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame({'Close': []})
        result = compute_indicators(empty_df)
        
        # Should return DataFrame with expected columns but no data
        expected_columns = [
            'Close', 'SMA_20', 'SMA_50', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        assert len(result) == 0
    
    def test_insufficient_data_for_indicators(self):
        """Test behavior when insufficient data for some indicators."""
        # Only 5 data points - insufficient for most indicators
        small_df = pd.DataFrame({'Close': [100, 101, 99, 102, 98]})
        result = compute_indicators(small_df)
        
        # Should have all columns but many NaN values
        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        assert result['SMA_20'].isna().all()  # All NaN due to insufficient data
        assert result['SMA_50'].isna().all()  # All NaN due to insufficient data
    
    def test_data_types(self, sample_data):
        """Test that output data types are numeric."""
        result = compute_indicators(sample_data)
        
        numeric_columns = [
            'SMA_20', 'SMA_50', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI'
        ]
        
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"Column {col} should be numeric"
    
    def test_rsi_extreme_values(self):
        """Test RSI calculation with extreme price movements."""
        # Create data with extreme movements
        extreme_data = pd.DataFrame({
            'Close': [100] + [110] * 20 + [90] * 20  # Strong uptrend then downtrend
        })
        
        result = compute_indicators(extreme_data)
        rsi_values = result['RSI'].dropna()
        
        # RSI should still be within bounds
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Should have some high RSI values during uptrend
        assert rsi_values.max() > 70  # Typical overbought threshold


class TestPlotPriceChart:
    """Test suite for the plot_price_chart function."""
    
    @pytest.fixture
    def sample_data_with_indicators(self):
        """Create sample data with indicators computed."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        base_price = 100
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
        
        return compute_indicators(df)
    
    @pytest.fixture
    def sample_data_no_date(self):
        """Create sample data without Date column."""
        df = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105] * 10
        })
        return compute_indicators(df)
    
    def test_plot_price_chart_returns_figure(self, sample_data_with_indicators):
        """Test that plot_price_chart returns a matplotlib figure."""
        fig = plot_price_chart('AAPL', sample_data_with_indicators)
        
        assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
        plt.close(fig)  # Clean up
    
    def test_plot_price_chart_with_date_column(self, sample_data_with_indicators):
        """Test plotting with Date column."""
        fig = plot_price_chart('AAPL', sample_data_with_indicators)
        
        ax = fig.get_axes()[0]
        
        # Check that title includes ticker
        assert 'AAPL' in ax.get_title()
        assert 'Stock Price Analysis' in ax.get_title()
        
        # Check that we have lines plotted
        lines = ax.get_lines()
        assert len(lines) > 0, "Should have at least one line plotted"
        
        plt.close(fig)
    
    def test_plot_price_chart_without_date_column(self, sample_data_no_date):
        """Test plotting without Date column (using index)."""
        fig = plot_price_chart('TEST', sample_data_no_date)
        
        ax = fig.get_axes()[0]
        
        # Check basic properties
        assert 'TEST' in ax.get_title()
        lines = ax.get_lines()
        assert len(lines) > 0, "Should have lines plotted"
        
        plt.close(fig)
    
    def test_plot_price_chart_has_close_price_line(self, sample_data_with_indicators):
        """Test that close price line is always plotted."""
        fig = plot_price_chart('AAPL', sample_data_with_indicators)
        
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        
        # Should have at least one line (close price)
        assert len(lines) >= 1
        
        # Check legend includes Close Price
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'Close Price' in legend_labels
        
        plt.close(fig)
    
    def test_plot_price_chart_has_sma_lines(self, sample_data_with_indicators):
        """Test that SMA lines are plotted when data is available."""
        fig = plot_price_chart('AAPL', sample_data_with_indicators)
        
        ax = fig.get_axes()[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        
        # Should have SMA lines in legend (data has 100 points, so SMAs should exist)
        assert '20-day SMA' in legend_labels
        assert '50-day SMA' in legend_labels
        
        plt.close(fig)
    
    def test_plot_price_chart_has_bollinger_bands(self, sample_data_with_indicators):
        """Test that Bollinger Bands are plotted when data is available."""
        fig = plot_price_chart('AAPL', sample_data_with_indicators)
        
        ax = fig.get_axes()[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        
        # Should have Bollinger Band elements
        assert 'Bollinger Upper' in legend_labels
        assert 'Bollinger Lower' in legend_labels
        
        # Check for filled area (Bollinger Bands shading)
        collections = ax.collections
        assert len(collections) > 0, "Should have filled areas for Bollinger Bands"
        
        plt.close(fig)
    
    def test_plot_price_chart_custom_figsize(self, sample_data_with_indicators):
        """Test that custom figure size is respected."""
        custom_size = (10, 6)
        fig = plot_price_chart('AAPL', sample_data_with_indicators, figsize=custom_size)
        
        # Check figure size (allow small tolerance for DPI differences)
        actual_size = fig.get_size_inches()
        assert abs(actual_size[0] - custom_size[0]) < 0.1
        assert abs(actual_size[1] - custom_size[1]) < 0.1
        
        plt.close(fig)
    
    def test_plot_price_chart_with_minimal_data(self):
        """Test plotting with minimal data (insufficient for some indicators)."""
        minimal_df = pd.DataFrame({'Close': [100, 101, 99, 102, 98]})
        df_with_indicators = compute_indicators(minimal_df)
        
        fig = plot_price_chart('TEST', df_with_indicators)
        
        ax = fig.get_axes()[0]
        
        # Should still have close price line
        lines = ax.get_lines()
        assert len(lines) >= 1
        
        # Should have title and basic elements
        assert 'TEST' in ax.get_title()
        
        plt.close(fig)
    
    def test_plot_price_chart_styling_elements(self, sample_data_with_indicators):
        """Test that chart has proper styling elements."""
        fig = plot_price_chart('AAPL', sample_data_with_indicators)
        
        ax = fig.get_axes()[0]
        
        # Check title
        assert ax.get_title() != ''
        assert 'AAPL' in ax.get_title()
        
        # Check axis labels
        assert ax.get_ylabel() != ''
        assert '$' in ax.get_ylabel()  # Should indicate price in dollars
        
        # Check grid
        assert ax.grid
        
        # Check legend
        legend = ax.get_legend()
        assert legend is not None
        
        # Check spines styling (top and right should be hidden)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        
        plt.close(fig)
    
    def test_plot_price_chart_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame({'Close': []})
        df_with_indicators = compute_indicators(empty_df)
        
        # Should not raise an exception
        fig = plot_price_chart('EMPTY', df_with_indicators)
        
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


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