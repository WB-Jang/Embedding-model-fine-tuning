# Embedding-model-fine-tuning

Project for fine-tuning various embedding models including BGE-M3, Snowflake-Arctic, and others.

## Features

- Fine-tune embedding models using sentence-transformers
- Support for multiple embedding models (BGE-M3, Snowflake-Arctic, etc.)
- Docker and DevContainer support for consistent development environment
- Poetry for dependency management
- GPU support for efficient training

## Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- Python 3.9+ (if running locally)
- Poetry 1.7+ (if running locally)

## Getting Started

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up -d

# Access the container
docker-compose exec embedding-finetune bash

# Inside the container, run training
poetry run python -m embedding_finetune.train
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t embedding-model-fine-tuning .

# Run the container
docker run --gpus all -it -v $(pwd):/workspace embedding-model-fine-tuning

# Inside the container, run training
poetry run python -m embedding_finetune.train
```

### Option 3: Using VS Code DevContainer

1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. Wait for the container to build and start
5. Run training: `poetry run python -m embedding_finetune.train`

### Option 4: Local Development

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run training
poetry run python -m embedding_finetune.train
```

## Project Structure

```
.
├── .devcontainer/          # DevContainer configuration
│   └── devcontainer.json
├── src/
│   └── embedding_finetune/ # Main package
│       ├── __init__.py
│       ├── train.py        # Training logic
│       └── utils.py        # Utility functions
├── tests/                  # Test suite
│   ├── test_train.py
│   └── test_utils.py
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── pyproject.toml          # Poetry configuration
└── README.md               # This file
```

## Usage

### Basic Training Example

```python
from embedding_finetune.train import EmbeddingFineTuner

# Sample training data: (text1, text2, similarity_score)
data = [
    ("This is a sample sentence", "This is another sentence", 0.8),
    ("Machine learning is fascinating", "Deep learning is interesting", 0.7),
]

# Initialize fine-tuner
fine_tuner = EmbeddingFineTuner(model_name="BAAI/bge-small-en-v1.5")

# Prepare and train
train_examples = fine_tuner.prepare_training_data(data)
fine_tuner.train(train_examples, epochs=3, output_path="./models/my_model")
```

### Loading Data from Files

```python
from embedding_finetune.utils import load_training_data_from_csv
from embedding_finetune.train import EmbeddingFineTuner

# Load data from CSV
data = load_training_data_from_csv("training_data.csv")

# Train
fine_tuner = EmbeddingFineTuner()
train_examples = fine_tuner.prepare_training_data(data)
fine_tuner.train(train_examples)
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=embedding_finetune

# Run specific test file
poetry run pytest tests/test_train.py
```

### Code Formatting

```bash
# Format code with black
poetry run black src/ tests/

# Sort imports
poetry run isort src/ tests/

# Check with flake8
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/
```

## Supported Models

- BAAI/bge-m3
- BAAI/bge-small-en-v1.5
- BAAI/bge-base-en-v1.5
- BAAI/bge-large-en-v1.5
- Snowflake/snowflake-arctic-embed-*
- sentence-transformers/all-MiniLM-L6-v2
- Any model from sentence-transformers

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
