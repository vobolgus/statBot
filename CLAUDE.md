# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run application: `python main.py`
- Install dependencies: `pip install -r requirements.txt`
- Linting: `python -m pylint *.py`
- Type checking: `python -m mypy *.py`

## Code Style
- Indentation: 4 spaces
- Line length: maximum 88 characters
- Imports: grouped by standard library, third-party, local packages
- Docstrings: required for all functions with triple quotes
- Type hints: recommended for function parameters and return values
- Naming: snake_case for variables/functions, UPPER_CASE for constants
- Error handling: use try/except with specific exceptions
- Logging: use the logger instead of print statements
- Variable naming: descriptive names preferred over abbreviations
- Comments: Add comments for complex logic only
- Keep code modular with functions focused on single responsibility

## Project Structure
- config.py: Configuration parameters
- db.py: Database operations
- analytics.py: Data analysis functions
- plotting.py: Data visualization
- bot.py: Telegram bot functionality
- main.py: Entry point