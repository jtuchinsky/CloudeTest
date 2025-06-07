import pytest
from claudetest.main import main
from io import StringIO
import sys


def test_main_output(capsys):
    """Test that main function produces expected output."""
    main()
    captured = capsys.readouterr()
    assert "Hello, World!" in captured.out
    assert "Welcome to your new Python app!" in captured.out