# Contributing to Embedding Model Fine-tuning

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Environment Setup

Please see [DEVELOPMENT.md](DEVELOPMENT.md) for detailed instructions on setting up your development environment using Docker, DevContainer, or local installation.

## Code Style and Standards

This project follows these coding standards:

### Python Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Use type hints where appropriate
- Maximum line length: 100 characters

### Running Code Quality Tools

Before submitting a PR, ensure all quality checks pass:

```bash
# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Lint code
poetry run flake8 src/ tests/

# Type check
poetry run mypy src/

# Run tests
poetry run pytest
```

## Testing

### Writing Tests
- Write tests for all new features
- Maintain or improve code coverage
- Use pytest fixtures for test setup
- Use descriptive test names that explain what is being tested

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_utils.py::test_load_training_data_from_csv

# Run with coverage
poetry run pytest --cov=embedding_finetune --cov-report=html
```

## Submitting Changes

### Pull Request Process

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for your changes
4. **Update documentation** if needed
5. **Run all quality checks** and ensure they pass
6. **Commit your changes** with clear, descriptive messages
7. **Push to your fork** and submit a pull request

### Commit Message Guidelines

Follow these conventions for commit messages:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: Add support for custom loss functions

Implement ability to specify custom loss functions for fine-tuning.
This allows users to experiment with different training objectives.

Closes #123
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write code following the style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run tests
poetry run pytest

# Check code style
poetry run flake8 src/ tests/
poetry run black --check src/ tests/
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: your feature description"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Docker and DevContainer Considerations

All code must work properly in:
- Docker containers
- VS Code DevContainers
- Local Poetry environments

Test your changes in multiple environments when possible.

## Adding Dependencies

When adding new dependencies:

1. **Add to pyproject.toml**:
   ```bash
   poetry add package-name
   ```

2. **For development dependencies**:
   ```bash
   poetry add --group dev package-name
   ```

3. **Update lock file**:
   ```bash
   poetry lock
   ```

4. **Test in Docker**:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## Documentation

### Code Documentation
- Use docstrings for all public functions, classes, and modules
- Follow Google-style docstring format
- Include type hints in function signatures

Example:
```python
def train_model(
    model_name: str,
    data: List[Tuple[str, str, float]],
    epochs: int = 3
) -> None:
    """Train an embedding model.

    Args:
        model_name: Name of the pre-trained model to fine-tune
        data: Training data as list of (text1, text2, similarity) tuples
        epochs: Number of training epochs

    Raises:
        ValueError: If data is empty or epochs is negative
    """
    pass
```

### README Updates
Update README.md when:
- Adding new features
- Changing installation instructions
- Modifying usage examples

## Reporting Issues

### Bug Reports
When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, Docker version)
- Error messages and stack traces

### Feature Requests
When requesting features, include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)

## Code Review Process

All submissions require review. The process:

1. Automated checks run (tests, linting, security)
2. Code review by maintainers
3. Address feedback and update PR
4. Once approved, PR will be merged

## Questions?

If you have questions:
- Check existing issues and documentation
- Open a new issue with the "question" label
- Be specific and provide context

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
