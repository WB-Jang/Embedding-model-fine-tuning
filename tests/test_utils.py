"""Test utility functions."""

import json
import csv
from embedding_finetune.utils import (
    load_training_data_from_csv,
    load_training_data_from_json,
    save_config,
    load_config,
)


def test_load_training_data_from_csv(tmp_path):
    """Test loading training data from CSV."""
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["text1", "text2", "score"])
        writer.writerow(["hello", "hi", "0.9"])
        writer.writerow(["world", "earth", "0.8"])

    data = load_training_data_from_csv(str(csv_path))
    assert len(data) == 2
    assert data[0] == ("hello", "hi", 0.9)
    assert data[1] == ("world", "earth", 0.8)


def test_load_training_data_from_json(tmp_path):
    """Test loading training data from JSON."""
    json_path = tmp_path / "test.json"
    with open(json_path, "w") as f:
        json.dump(
            [
                {"text1": "hello", "text2": "hi", "score": 0.9},
                {"text1": "world", "text2": "earth", "score": 0.8},
            ],
            f,
        )

    data = load_training_data_from_json(str(json_path))
    assert len(data) == 2
    assert data[0] == ("hello", "hi", 0.9)
    assert data[1] == ("world", "earth", 0.8)


def test_save_and_load_config(tmp_path):
    """Test saving and loading configuration."""
    config = {
        "model_name": "BAAI/bge-m3",
        "batch_size": 16,
        "epochs": 3,
    }

    config_path = tmp_path / "config.json"
    save_config(config, str(config_path))

    loaded_config = load_config(str(config_path))
    assert loaded_config == config
