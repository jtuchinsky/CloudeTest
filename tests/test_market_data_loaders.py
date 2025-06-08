import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from claudetest.pdf_report.market_data_loaders import load_market_data_from_yahoo


class TestLoadMarketDataFromYahoo:
    """Test suite for the load_market_data_from_yahoo function."""

    @pytest.fixture
    def mock_yahoo_data(self):
        """Create mock Yahoo Finance data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')  # ~1 year of trading days
        prices = np.random.uniform(90, 110, len(dates))  # Random prices between 90 - 110

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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

    @patch('claudetest.pdf_report.market_data_loaders.yf.Ticker')
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

