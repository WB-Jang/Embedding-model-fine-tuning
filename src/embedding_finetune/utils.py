"""Utility functions for embedding model fine-tuning."""

import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_training_data_from_csv(
    csv_path: str, text1_col: str = "text1", text2_col: str = "text2", score_col: str = "score"
) -> List[Tuple[str, str, float]]:
    """Load training data from CSV file.

    Args:
        csv_path: Path to CSV file
        text1_col: Column name for first text
        text2_col: Column name for second text
        score_col: Column name for similarity score

    Returns:
        List of tuples (text1, text2, score)
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((row[text1_col], row[text2_col], float(row[score_col])))
    logger.info(f"Loaded {len(data)} training examples from {csv_path}")
    return data


def load_training_data_from_json(json_path: str) -> List[Tuple[str, str, float]]:
    """Load training data from JSON file.

    Expected format:
    [
        {"text1": "...", "text2": "...", "score": 0.8},
        ...
    ]

    Args:
        json_path: Path to JSON file

    Returns:
        List of tuples (text1, text2, score)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = [(item["text1"], item["text2"], float(item["score"])) for item in raw_data]
    logger.info(f"Loaded {len(data)} training examples from {json_path}")
    return data


def save_config(config: Dict[str, Any], output_path: str):
    """Save training configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config
