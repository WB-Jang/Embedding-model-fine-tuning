"""Test training module."""

from embedding_finetune.train import EmbeddingFineTuner


def test_embedding_fine_tuner_initialization():
    """Test that EmbeddingFineTuner can be initialized."""
    # Use a small model for testing
    fine_tuner = EmbeddingFineTuner(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert fine_tuner.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert fine_tuner.model is not None
    assert fine_tuner.use_asymmetric_loss is False


def test_embedding_fine_tuner_asymmetric_initialization():
    """Test that EmbeddingFineTuner can be initialized with asymmetric loss."""
    fine_tuner = EmbeddingFineTuner(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_asymmetric_loss=True
    )
    assert fine_tuner.use_asymmetric_loss is True


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


def test_prepare_term_definition_data():
    """Test preparing term-definition data."""
    fine_tuner = EmbeddingFineTuner(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_asymmetric_loss=True
    )

    term_definition_data = [
        ("GDP", "Gross Domestic Product is the total value of goods produced."),
        ("인플레이션", "물가가 지속적으로 상승하는 경제 현상"),
    ]

    examples = fine_tuner.prepare_term_definition_data(term_definition_data)
    assert len(examples) == 2
    assert examples[0].texts == [
        "GDP", "Gross Domestic Product is the total value of goods produced."
    ]
    assert examples[1].texts == ["인플레이션", "물가가 지속적으로 상승하는 경제 현상"]
    # Term-definition pairs don't have labels
    assert not hasattr(examples[0], 'label') or examples[0].label is None


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
