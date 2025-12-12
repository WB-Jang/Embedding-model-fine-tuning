# Embedding-model-fine-tuning
Project doing fine-tuning process on various embedding models including bge-m3, snowflake-arctic etc.

Google Colab으로 테스트 완료, 주피터노트북 src에서 참조

## 📋 Requirements

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ VRAM for RTX 4060 or similar GPUs

## 🚀 Quick Start

### Option 1: Using Poetry (Recommended for Development)

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Run the training script:
```bash
poetry run python src/main.py
```

### Option 2: Using pip and Virtual Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python src/main.py
```

### Option 3: Using Docker

1. Build the Docker image:
```bash
docker build -t embedding-finetuning .
```

2. Run the container:
```bash
docker run -v $(pwd)/finetuned_finance_model:/app/finetuned_finance_model embedding-finetuning
```

Or use Docker Compose:
```bash
docker-compose up --build
```

**Note**: For GPU support in Docker, you need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and uncomment the GPU section in `docker-compose.yml`.

### Option 4: Using VS Code Dev Container

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
2. Open the project folder in VS Code
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. Once the container is built, open a terminal and run:
```bash
python src/main.py
```

## 📁 Project Structure

```
.
├── src/
│   └── main.py           # Main training script
├── .devcontainer/        # VS Code dev container configuration
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── pyproject.toml        # Poetry configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

The training script (`src/main.py`) includes several configurable parameters:

- `MODEL_ID`: The base model to fine-tune (default: "BAAI/bge-m3")
- `OUTPUT_PATH`: Where to save the fine-tuned model (default: "./finetuned_finance_model")
- `BATCH_SIZE`: Training batch size (default: 4, adjust based on your GPU memory)
- `NUM_EPOCHS`: Number of training epochs (default: 3)

## 📊 Data

The script currently uses dummy data for testing. To use your own data:

1. Prepare a CSV file with columns: `term` and `definition`
2. Update the data loading section in `src/main.py`:
```python
df = pd.read_csv("your_data.csv")
```

## 🎯 Model Training

The script performs fine-tuning using:
- **Model**: BAAI/bge-m3 (multilingual embedding model)
- **Loss Function**: MultipleNegativesRankingLoss
- **Data Augmentation**: Multiple query variations per term
- **Validation**: Built-in similarity testing

## 📝 Output

After training, the fine-tuned model will be saved to `./finetuned_finance_model/` directory. You can load and use it with:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("./finetuned_finance_model")
embeddings = model.encode(["your text here"])
```

## ⚠️ Troubleshooting

- **Out of Memory (OOM)**: Reduce `BATCH_SIZE` in `src/main.py`
- **Slow Training**: Ensure CUDA is available: `torch.cuda.is_available()`
- **Dependencies Issues**: Try updating pip: `pip install --upgrade pip`

## 📄 License

This project is available for educational and research purposes.
