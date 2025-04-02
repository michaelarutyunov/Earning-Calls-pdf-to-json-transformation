# Contributing to QEC PDF to JSON Transformation

Thank you for your interest in contributing to this project! This document provides guidelines and steps for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/project_EC_pdf_formatting.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

## Making Changes

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Update documentation as needed
5. Commit your changes with clear commit messages
6. Push to your fork
7. Create a Pull Request

## Testing

- Run tests before submitting PR: `python -m pytest tests/`
- Ensure all tests pass
- Add new tests for new functionality

## Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update CHANGELOG.md with your changes

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the CHANGELOG.md with your changes
3. The PR will be merged once you have the sign-off of at least one other developer

## Questions?

Feel free to open an issue for any questions or concerns. 