import pytest
from unittest.mock import patch, MagicMock
from claudetest.main import main, get_tickers, open_pdf
import platform


class TestGetTickers:
    """Test suite for the get_tickers function."""
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_single_ticker(self, mock_print, mock_input):
        """Test collecting a single ticker."""
        mock_input.side_effect = ['AAPL', 'done'] + [''] * 10
        
        result = get_tickers()
        
        assert result == ['AAPL']
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_multiple_tickers(self, mock_print, mock_input):
        """Test collecting multiple tickers."""
        mock_input.side_effect = ['AAPL', 'MSFT', 'GOOGL', 'done'] + [''] * 10
        
        result = get_tickers()
        
        assert result == ['AAPL', 'MSFT', 'GOOGL']
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_quit(self, mock_print, mock_input):
        """Test quitting without entering tickers."""
        mock_input.return_value = 'quit'
        
        result = get_tickers()
        
        assert result is None
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_max_five(self, mock_print, mock_input):
        """Test that maximum 5 tickers are collected."""
        mock_input.side_effect = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        result = get_tickers()
        
        assert len(result) == 5
        assert result == ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_duplicate_handling(self, mock_print, mock_input):
        """Test that duplicate tickers are rejected."""
        mock_input.side_effect = ['AAPL', 'AAPL', 'MSFT', 'done'] + [''] * 10
        
        result = get_tickers()
        
        assert result == ['AAPL', 'MSFT']
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_case_insensitive(self, mock_print, mock_input):
        """Test that tickers are converted to uppercase."""
        mock_input.side_effect = ['aapl', 'msft', 'done'] + [''] * 10
        
        result = get_tickers()
        
        assert result == ['AAPL', 'MSFT']
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_invalid_format(self, mock_print, mock_input):
        """Test that invalid ticker formats are rejected."""
        mock_input.side_effect = ['123', 'AAPL123', 'VERYLONGTICKER', 'AAPL', 'done'] + [''] * 10
        
        result = get_tickers()
        
        assert result == ['AAPL']
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_tickers_empty_required(self, mock_print, mock_input):
        """Test that first ticker is required."""
        mock_input.side_effect = ['', 'AAPL', 'done'] + [''] * 10  # Add extra empty strings
        
        result = get_tickers()
        
        assert result == ['AAPL']


class TestOpenPdf:
    """Test suite for the open_pdf function."""
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_open_pdf_macos(self, mock_system, mock_run):
        """Test PDF opening on macOS."""
        mock_system.return_value = "Darwin"
        
        open_pdf("test.pdf")
        
        mock_run.assert_called_once_with(["open", "test.pdf"], check=True)
    
    @patch('claudetest.main.os.startfile', create=True)
    @patch('platform.system')
    def test_open_pdf_windows(self, mock_system, mock_startfile):
        """Test PDF opening on Windows."""
        mock_system.return_value = "Windows"
        
        open_pdf("test.pdf")
        
        mock_startfile.assert_called_once_with("test.pdf")
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_open_pdf_linux(self, mock_system, mock_run):
        """Test PDF opening on Linux."""
        mock_system.return_value = "Linux"
        
        open_pdf("test.pdf")
        
        mock_run.assert_called_once_with(["xdg-open", "test.pdf"], check=True)
    
    @patch('platform.system')
    def test_open_pdf_unsupported_system(self, mock_system, capsys):
        """Test PDF opening on unsupported system."""
        mock_system.return_value = "UnknownOS"
        
        open_pdf("test.pdf")
        
        captured = capsys.readouterr()
        assert "Cannot automatically open PDF on UnknownOS" in captured.out
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_open_pdf_error_handling(self, mock_system, mock_run, capsys):
        """Test error handling when PDF opening fails."""
        mock_system.return_value = "Darwin"
        mock_run.side_effect = Exception("Command failed")
        
        open_pdf("test.pdf")
        
        captured = capsys.readouterr()
        assert "Could not open PDF automatically" in captured.out


class TestMain:
    """Test suite for the main function."""
    
    @patch('claudetest.main.open_pdf')
    @patch('claudetest.main.generate_finance_report')
    @patch('claudetest.main.get_tickers')
    @patch('builtins.input')
    def test_main_successful_flow(self, mock_input, mock_get_tickers, mock_generate_report, mock_open_pdf):
        """Test successful main function execution."""
        # Setup mocks
        mock_get_tickers.return_value = ['AAPL', 'MSFT']
        mock_generate_report.return_value = "financial_report.pdf"
        mock_input.return_value = 'y'  # User wants to open PDF
        
        # Test function
        main()
        
        # Verify calls
        mock_get_tickers.assert_called_once()
        mock_generate_report.assert_called_once_with(['AAPL', 'MSFT'])
        mock_open_pdf.assert_called_once_with("financial_report.pdf")
    
    @patch('claudetest.main.get_tickers')
    def test_main_quit_early(self, mock_get_tickers):
        """Test main function when user quits early."""
        mock_get_tickers.return_value = None
        
        # Should not raise any exceptions
        main()
        
        mock_get_tickers.assert_called_once()
    
    @patch('claudetest.main.generate_finance_report')
    @patch('claudetest.main.get_tickers')
    @patch('builtins.input')
    def test_main_decline_open_pdf(self, mock_input, mock_get_tickers, mock_generate_report, capsys):
        """Test main function when user declines to open PDF."""
        # Setup mocks
        mock_get_tickers.return_value = ['AAPL']
        mock_generate_report.return_value = "financial_report.pdf"
        mock_input.return_value = 'n'  # User doesn't want to open PDF
        
        # Test function
        main()
        
        # Verify output
        captured = capsys.readouterr()
        assert "Report saved successfully" in captured.out
    
    @patch('claudetest.main.generate_finance_report')
    @patch('claudetest.main.get_tickers')
    def test_main_keyboard_interrupt(self, mock_get_tickers, mock_generate_report):
        """Test main function with keyboard interrupt."""
        mock_get_tickers.side_effect = KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
    
    @patch('claudetest.main.generate_finance_report')
    @patch('claudetest.main.get_tickers')
    def test_main_exception_handling(self, mock_get_tickers, mock_generate_report):
        """Test main function exception handling."""
        mock_get_tickers.return_value = ['AAPL']
        mock_generate_report.side_effect = Exception("Network error")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1