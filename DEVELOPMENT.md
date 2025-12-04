# Development Environment Setup

This document describes how to set up and use the development environment for the Embedding Model Fine-tuning project.

## Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- VS Code with Dev Containers extension (for DevContainer development)
- Git

## Option 1: Using Docker Compose (Recommended for Development)

Docker Compose provides the easiest way to get started with the project.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WB-Jang/Embedding-model-fine-tuning.git
   cd Embedding-model-fine-tuning
   ```

2. **Build and start the container:**
   ```bash
   docker-compose up -d
   ```

3. **Access the container:**
   ```bash
   docker-compose exec embedding-finetune bash
   ```

4. **Run your training code:**
   ```bash
   poetry run python -m embedding_finetune.train
   ```

5. **Stop the container:**
   ```bash
   docker-compose down
   ```

## Option 2: Using VS Code DevContainer

DevContainers provide a seamless development experience within VS Code.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WB-Jang/Embedding-model-fine-tuning.git
   ```

2. **Open in VS Code:**
   ```bash
   cd Embedding-model-fine-tuning
   code .
   ```

3. **Reopen in Container:**
   - Press `F1` or `Ctrl+Shift+P`
   - Select "Dev Containers: Reopen in Container"
   - Wait for the container to build (first time only)

4. **Development:**
   - All dependencies are automatically installed
   - Python interpreter is pre-configured
   - Extensions are automatically installed
   - You can run, debug, and test code directly

## Option 3: Using Docker Directly

For more control over the container execution.

### Steps

1. **Build the image:**
   ```bash
   docker build -t embedding-model-fine-tuning .
   ```

2. **Run the container:**
   ```bash
   docker run --gpus all -it -v $(pwd):/workspace embedding-model-fine-tuning
   ```

3. **Inside the container:**
   ```bash
   poetry install
   poetry run python -m embedding_finetune.train
   ```

## Option 4: Local Development (Without Docker)

For development without containerization.

### Steps

1. **Install Poetry:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Run code:**
   ```bash
   poetry run python -m embedding_finetune.train
   ```

## Running Tests

Inside the container or local environment:

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_utils.py

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=embedding_finetune
```

## Code Quality Tools

### Linting with Flake8
```bash
poetry run flake8 src/ tests/
```

### Formatting with Black
```bash
poetry run black src/ tests/
```

### Import Sorting with isort
```bash
poetry run isort src/ tests/
```

### Type Checking with mypy
```bash
poetry run mypy src/
```

## Project Structure

```
.
├── .devcontainer/          # VS Code DevContainer configuration
│   └── devcontainer.json
├── src/
│   └── embedding_finetune/ # Main package
│       ├── __init__.py
│       ├── train.py        # Training module
│       └── utils.py        # Utility functions
├── tests/                  # Test suite
│   ├── test_train.py
│   └── test_utils.py
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── pyproject.toml          # Poetry configuration
├── poetry.lock             # Locked dependencies
└── README.md              # Main documentation
```

## GPU Support

All Docker configurations include GPU support via NVIDIA Container Toolkit. Ensure:

1. NVIDIA drivers are installed on the host
2. NVIDIA Container Toolkit is installed
3. Docker has access to GPU devices

Test GPU availability:
```bash
docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

## Environment Variables

You can customize the environment by setting these variables:

- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `POETRY_VIRTUALENVS_IN_PROJECT=true` - Create venv in project directory
- `CUDA_VISIBLE_DEVICES` - Control which GPUs to use

## Troubleshooting

### Container fails to start
- Ensure Docker is running
- Check that ports are not already in use
- Verify NVIDIA drivers and toolkit are installed

### Poetry dependencies fail to install
- Check network connectivity
- Try clearing Poetry cache: `poetry cache clear pypi --all`
- Manually update lock file: `poetry lock --no-update`

### GPU not available in container
- Verify NVIDIA Container Toolkit: `docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
- Check Docker daemon configuration for GPU support
- Ensure `--gpus=all` flag is used when running containers

## Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
