"""Test utility functions."""

import json
import csv
import tempfile
from pathlib import Path
from embedding_finetune.utils import (
    load_training_data_from_csv,
    load_training_data_from_json,
    save_config,
    load_config,
)


def test_load_training_data_from_csv():
    """Test loading training data from CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["text1", "text2", "score"])
        writer.writerow(["hello", "hi", "0.9"])
        writer.writerow(["world", "earth", "0.8"])
        csv_path = f.name

    try:
        data = load_training_data_from_csv(csv_path)
        assert len(data) == 2
        assert data[0] == ("hello", "hi", 0.9)
        assert data[1] == ("world", "earth", 0.8)
    finally:
        Path(csv_path).unlink()


def test_load_training_data_from_json():
    """Test loading training data from JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            [
                {"text1": "hello", "text2": "hi", "score": 0.9},
                {"text1": "world", "text2": "earth", "score": 0.8},
            ],
            f,
        )
        json_path = f.name

    try:
        data = load_training_data_from_json(json_path)
        assert len(data) == 2
        assert data[0] == ("hello", "hi", 0.9)
        assert data[1] == ("world", "earth", 0.8)
    finally:
        Path(json_path).unlink()


def test_save_and_load_config():
    """Test saving and loading configuration."""
    config = {
        "model_name": "BAAI/bge-m3",
        "batch_size": 16,
        "epochs": 3,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        save_config(config, str(config_path))

        loaded_config = load_config(str(config_path))
        assert loaded_config == config
