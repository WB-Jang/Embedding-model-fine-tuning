# Embedding-model-fine-tuning

Project for fine-tuning various embedding models including BGE-M3, Snowflake-Arctic, and others.

## Features

- Fine-tune embedding models using sentence-transformers
- Support for multiple embedding models (BGE-M3, Snowflake-Arctic, etc.)
- **Two training modes:**
  - **Symmetric:** For similarity tasks with (text1, text2, similarity_score) format
  - **Asymmetric:** For term-definition pairs like dictionary data (term, definition)
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

### Basic Training Example (Symmetric Similarity)

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

### Term-Definition Training (Asymmetric)

For dictionary data like the 기획재정부 시사경제용어사전 (Ministry of Economy and Finance Current Economic Terms Dictionary), use asymmetric training:

```python
from embedding_finetune.train import EmbeddingFineTuner

# Dictionary data: (term, definition)
dictionary_data = [
    ("GDP", "국내총생산은 일정 기간 동안 한 나라 안에서 생산된 모든 최종 재화와 서비스의 시장 가치를 합한 것이다."),
    ("인플레이션", "물가가 지속적으로 상승하는 경제 현상으로, 화폐 가치가 하락하고 구매력이 감소하는 것을 의미한다."),
    ("양적완화", "중앙은행이 국채나 회사채 등을 매입하여 시중에 유동성을 공급함으로써 경기를 부양하는 통화정책이다."),
]

# Initialize fine-tuner with asymmetric loss
fine_tuner = EmbeddingFineTuner(
    model_name="BAAI/bge-m3",  # Multilingual model for Korean
    use_asymmetric_loss=True
)

# Prepare and train
train_examples = fine_tuner.prepare_term_definition_data(dictionary_data)
fine_tuner.train(train_examples, epochs=10, output_path="./models/economic_terms_model")
```

You can also run the example script:

```bash
poetry run python src/main.py
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
