# Contributing to Document Search Engine

Thank you for your interest in contributing to the Document Search Engine project! This document provides guidelines and instructions for contributing.

## Table of Contents

* [Code of Conduct](#code-of-conduct)
* [Getting Started](#getting-started)
* [Development Setup](#development-setup)
* [Making Changes](#making-changes)
* [Coding Standards](#coding-standards)
* [Testing](#testing)
* [Submitting Changes](#submitting-changes)
* [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment.

### Our Standards

* Be respectful and inclusive
* Accept constructive criticism gracefully
* Focus on what's best for the project
* Show empathy towards other contributors

## Getting Started

### Prerequisites

* Python 3.8 or higher
* Git for version control
* Basic understanding of search engines and text processing

### Areas for Contribution

We welcome contributions in the following areas:

* **Bug fixes**: Fix issues reported in GitHub Issues
* **New features**: Implement features from the roadmap
* **Documentation**: Improve or add documentation
* **Tests**: Add or improve test coverage
* **Performance**: Optimize indexing or search performance
* **New document formats**: Add support for additional file types

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/document-search-engine.git
cd document-search-engine
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]
```

### 4. Verify Installation

```bash
# Run tests to verify setup
pytest tests/

# Run example scripts
python examples/basic_usage.py
```

## Making Changes

### 1. Create a Branch

Create a feature branch for your changes:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Branch Naming Convention

* `feature/` - New features
* `fix/` - Bug fixes
* `docs/` - Documentation changes
* `test/` - Test additions or modifications
* `refactor/` - Code refactoring

### 2. Make Your Changes

* Write clear, readable code
* Follow the coding standards (see below)
* Add tests for new functionality
* Update documentation as needed
* Keep commits focused and atomic

### 3. Commit Messages

Write clear, descriptive commit messages:

```
Type: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
```

**Types:**
* `feat:` - New feature
* `fix:` - Bug fix
* `docs:` - Documentation only
* `test:` - Adding or updating tests
* `refactor:` - Code refactoring
* `perf:` - Performance improvements
* `style:` - Code style changes (formatting)

**Examples:**
```
feat: Add support for RTF documents

fix: Correct phrase search position tracking

docs: Update README with new installation steps
```

## Coding Standards

### Python Style Guide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
* Use 4 spaces for indentation (no tabs)
* Maximum line length: 100 characters
* Use meaningful variable and function names

### Code Formatting

We use `black` for automatic code formatting:

```bash
# Format your code
black src/ tests/ examples/

# Check formatting
black --check src/ tests/ examples/
```

### Linting

Use `flake8` for linting:

```bash
flake8 src/ tests/ examples/
```

### Type Hints

Use type hints for function parameters and return values:

```python
def search(self, query: str, top_k: int = 10) -> List[Dict]:
    """Search for documents matching the query.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of matching documents with scores
    """
    pass
```

### Documentation

* Add docstrings to all public classes and methods
* Use Google-style docstrings
* Include examples in docstrings when helpful

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> example_function("test", 5)
        True
    """
    pass
```

## Testing

### Writing Tests

* Write tests for all new functionality
* Aim for high test coverage (>80%)
* Use pytest fixtures for common setup
* Test both success and failure cases

### Test Structure

```python
import pytest

class TestYourFeature:
    """Test suite for your feature."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {"key": "value"}
    
    def test_basic_functionality(self, setup_data):
        """Test basic functionality."""
        result = your_function(setup_data)
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_search_engine.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Test Coverage

Check test coverage:

```bash
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html to view coverage report
```

## Submitting Changes

### 1. Update Your Branch

```bash
# Get latest changes from main
git checkout main
git pull origin main

# Rebase your branch
git checkout your-branch
git rebase main
```

### 2. Run Final Checks

Before submitting, ensure:

```bash
# Format code
black src/ tests/ examples/

# Run linter
flake8 src/ tests/ examples/

# Run all tests
pytest tests/

# Check type hints
mypy src/
```

### 3. Push Your Changes

```bash
git push origin your-branch
```

### 4. Create Pull Request

* Go to GitHub and create a pull request
* Use a clear, descriptive title
* Describe what changes you made and why
* Reference any related issues
* Request review from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Reporting Issues

### Bug Reports

When reporting bugs, include:

* Clear, descriptive title
* Steps to reproduce
* Expected behavior
* Actual behavior
* Python version and environment details
* Relevant code snippets or error messages

### Feature Requests

When requesting features, include:

* Clear description of the feature
* Use case and motivation
* Possible implementation approach
* Any relevant examples

### Issue Template

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- OS:
- Package version:

## Additional Context
Any other relevant information
```

## Development Best Practices

### Code Review

* Be open to feedback
* Respond to review comments promptly
* Make requested changes in new commits
* Ask questions if unclear

### Performance Considerations

* Profile code for performance bottlenecks
* Use appropriate data structures
* Consider memory usage for large documents
* Add benchmarks for critical paths

### Security

* Validate all user inputs
* Handle file operations safely
* Don't commit sensitive information
* Use secure dependencies

### Documentation

* Update README for new features
* Add examples for complex functionality
* Keep API documentation current
* Document breaking changes

## Questions?

If you have questions about contributing:

* Check existing documentation
* Review closed issues and PRs
* Ask in GitHub Discussions
* Contact maintainers

## Recognition

Contributors will be:

* Listed in CONTRIBUTORS.md
* Mentioned in release notes
* Credited in documentation

Thank you for contributing to Document Search Engine! ✨