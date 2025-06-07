# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- **Install in development mode**: `pip install -e .`
- **Run the application**: `python -m claudetest.main` or `claudetest` (after install)
- **Run tests**: `pytest tests/`
- **Install dependencies**: `pip install -r requirements.txt`

## Project Structure

This is a Python package with a proper directory structure following Python packaging best practices.

- `src/claudetest/`: Main package source code
  - `main.py`: Application entry point with primary execution logic
  - `__init__.py`: Package initialization
- `tests/`: Test files using pytest
- `docs/`: Documentation directory
- `pyproject.toml`: Modern Python packaging configuration
- `requirements.txt`: Runtime dependencies