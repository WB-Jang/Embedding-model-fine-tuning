"""Test training module."""

from embedding_finetune.train import EmbeddingFineTuner


def test_embedding_fine_tuner_initialization():
    """Test that EmbeddingFineTuner can be initialized."""
    # Use a small model for testing
    fine_tuner = EmbeddingFineTuner(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert fine_tuner.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert fine_tuner.model is not None


def test_prepare_training_data():
    """Test preparing training data."""
    fine_tuner = EmbeddingFineTuner(model_name="sentence-transformers/all-MiniLM-L6-v2")

    sample_data = [
        ("text1", "text2", 0.8),
        ("text3", "text4", 0.6),
    ]

    examples = fine_tuner.prepare_training_data(sample_data)
    assert len(examples) == 2
    assert examples[0].texts == ["text1", "text2"]
    assert examples[0].label == 0.8


def test_encode():
    """Test encoding texts to embeddings."""
    fine_tuner = EmbeddingFineTuner(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = ["This is a test", "Another test"]
    embeddings = fine_tuner.encode(texts)

    assert embeddings.shape[0] == 2  # Two texts
    assert embeddings.shape[1] > 0  # Embedding dimension


def test_train_validation():
    """Test that train method validates input parameters."""
    import pytest
    fine_tuner = EmbeddingFineTuner(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Test empty train_examples
    with pytest.raises(ValueError, match="train_examples cannot be empty"):
        fine_tuner.train([])

    # Test invalid batch_size
    sample_data = [("text1", "text2", 0.8)]
    examples = fine_tuner.prepare_training_data(sample_data)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        fine_tuner.train(examples, batch_size=0)

    # Test invalid epochs
    with pytest.raises(ValueError, match="epochs must be positive"):
        fine_tuner.train(examples, epochs=-1)

    # Test empty output_path
    with pytest.raises(ValueError, match="output_path cannot be empty"):
        fine_tuner.train(examples, output_path="")
