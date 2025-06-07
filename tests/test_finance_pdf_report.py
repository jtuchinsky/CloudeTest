import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from claudetest.pdf_report.finance_pdf_report import compute_indicators, plot_price_chart, load_market_data_from_yahoo


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


class TestLoadMarketDataFromYahoo:
    """Test suite for the load_market_data_from_yahoo function."""
    
    @pytest.fixture
    def mock_yahoo_data(self):
        """Create mock Yahoo Finance data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')  # ~1 year of trading days
        prices = np.random.uniform(90, 110, len(dates))  # Random prices between 90-110
        
        # Create mock history DataFrame with Yahoo Finance structure
        mock_hist = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        mock_hist.index.name = 'Date'
        
        return mock_hist
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_success(self, mock_ticker, mock_yahoo_data):
        """Test successful data loading from Yahoo Finance."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_yahoo_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function
        result = load_market_data_from_yahoo('AAPL')
        
        # Verify function was called correctly
        mock_ticker.assert_called_once_with('AAPL')
        mock_ticker_instance.history.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert 'Date' in result.columns
        assert 'Close' in result.columns
        assert len(result.columns) == 2
        
        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        assert pd.api.types.is_numeric_dtype(result['Close'])
        
        # Verify data is sorted by date (oldest first)
        assert result['Date'].is_monotonic_increasing
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_date_range(self, mock_ticker, mock_yahoo_data):
        """Test that function requests correct date range (past year)."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_yahoo_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Capture current time for comparison
        before_call = datetime.now()
        
        # Test function
        load_market_data_from_yahoo('AAPL')
        
        after_call = datetime.now()
        
        # Verify history was called with date range
        call_args = mock_ticker_instance.history.call_args
        assert 'start' in call_args.kwargs
        assert 'end' in call_args.kwargs
        
        start_date = call_args.kwargs['start']
        end_date = call_args.kwargs['end']
        
        # Verify date range is approximately 1 year
        date_diff = (end_date - start_date).days
        assert 360 <= date_diff <= 370  # Allow some flexibility
        
        # Verify end date is close to current time
        assert before_call <= end_date <= after_call + timedelta(seconds=1)
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_empty_response(self, mock_ticker):
        """Test handling of empty response from Yahoo Finance."""
        # Setup mock to return empty DataFrame
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_market_data_from_yahoo('INVALID')
        
        assert "No data available for ticker 'INVALID'" in str(exc_info.value)
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_invalid_ticker(self, mock_ticker):
        """Test handling of invalid ticker symbols."""
        # Setup mock to raise exception
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("No data found, symbol may be delisted")
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_market_data_from_yahoo('INVALID123')
        
        assert "Invalid ticker symbol 'INVALID123'" in str(exc_info.value)
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_network_error(self, mock_ticker):
        """Test handling of network/connection errors."""
        # Setup mock to raise network exception
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("Connection timeout")
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function should raise general Exception
        with pytest.raises(Exception) as exc_info:
            load_market_data_from_yahoo('AAPL')
        
        assert "Error fetching data for ticker 'AAPL'" in str(exc_info.value)
        assert "Connection timeout" in str(exc_info.value)
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_data_structure(self, mock_ticker, mock_yahoo_data):
        """Test that returned data has correct structure and content."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_yahoo_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function
        result = load_market_data_from_yahoo('AAPL')
        
        # Verify data structure
        assert len(result) > 0, "Should return non-empty DataFrame"
        assert len(result) == len(mock_yahoo_data), "Should return all data points"
        
        # Verify Close prices are within reasonable range
        assert result['Close'].min() > 0, "Close prices should be positive"
        assert not result['Close'].isna().any(), "Should not have NaN values"
        
        # Verify dates are unique and in ascending order
        assert result['Date'].is_unique, "Dates should be unique"
        assert result['Date'].is_monotonic_increasing, "Dates should be in ascending order"
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_ticker_case_handling(self, mock_ticker, mock_yahoo_data):
        """Test that function handles ticker case correctly."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_yahoo_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test with lowercase ticker
        result = load_market_data_from_yahoo('aapl')
        
        # Verify ticker was passed as-is to yfinance (yfinance handles case)
        mock_ticker.assert_called_once_with('aapl')
        assert isinstance(result, pd.DataFrame)
    
    @patch('claudetest.pdf_report.finance_pdf_report.yf.Ticker')
    def test_load_market_data_integration_with_compute_indicators(self, mock_ticker, mock_yahoo_data):
        """Test that loaded data works with compute_indicators function."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_yahoo_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Load data and compute indicators
        market_data = load_market_data_from_yahoo('AAPL')
        indicators_data = compute_indicators(market_data)
        
        # Verify indicators were computed successfully
        expected_columns = [
            'Date', 'Close', 'SMA_20', 'SMA_50', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI'
        ]
        
        for col in expected_columns:
            assert col in indicators_data.columns, f"Column {col} should be present"
        
        # Verify original data is preserved
        pd.testing.assert_series_equal(
            market_data['Close'], 
            indicators_data['Close'], 
            check_names=False
        )