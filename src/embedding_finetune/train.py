"""Main training module for fine-tuning embedding models."""

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingFineTuner:
    """Fine-tune embedding models like BGE-M3, Snowflake-Arctic, etc.

    Supports both symmetric (text1, text2, similarity) and asymmetric (term, definition)
    training scenarios.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        use_asymmetric_loss: bool = False,
    ):
        """Initialize the fine-tuner.

        Args:
            model_name: Name of the pre-trained model to fine-tune
            device: Device to use for training (cuda/cpu)
            use_asymmetric_loss: If True, use MultipleNegativesRankingLoss for
                asymmetric tasks like term-definition matching. If False, use
                CosineSimilarityLoss for symmetric similarity tasks.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_asymmetric_loss = use_asymmetric_loss
        logger.info(f"Initializing model: {model_name} on device: {self.device}")
        logger.info(f"Using asymmetric loss: {use_asymmetric_loss}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def prepare_training_data(
        self, data: List[Tuple[str, str, float]]
    ) -> List[InputExample]:
        """Prepare training data for fine-tuning (symmetric similarity task).

        Args:
            data: List of tuples (text1, text2, similarity_score)

        Returns:
            List of InputExample objects
        """
        examples = []
        for text1, text2, score in data:
            examples.append(InputExample(texts=[text1, text2], label=score))
        return examples

    def prepare_term_definition_data(
        self, data: List[Tuple[str, str]]
    ) -> List[InputExample]:
        """Prepare term-definition pairs for fine-tuning (asymmetric task).

        This is suitable for dictionary-like data where terms (short queries)
        need to match with definitions (longer documents).

        Args:
            data: List of tuples (term, definition)

        Returns:
            List of InputExample objects without labels (for MultipleNegativesRankingLoss)
        """
        examples = []
        for term, definition in data:
            # For asymmetric tasks, we don't need labels
            # MultipleNegativesRankingLoss will use in-batch negatives
            examples.append(InputExample(texts=[term, definition]))
        logger.info(f"Prepared {len(examples)} term-definition pairs")
        return examples

    def train(
        self,
        train_examples: List[InputExample],
        batch_size: int = 16,
        epochs: int = 3,
        output_path: str = "./models/fine_tuned_model",
        warmup_steps: int = 100,
    ):
        """Fine-tune the embedding model.

        Args:
            train_examples: Training data as InputExample objects
            batch_size: Batch size for training
            epochs: Number of training epochs
            output_path: Path to save the fine-tuned model
            warmup_steps: Number of warmup steps

        Raises:
            ValueError: If input parameters are invalid
        """
        if not train_examples:
            raise ValueError("train_examples cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not output_path:
            raise ValueError("output_path cannot be empty")

        logger.info(f"Starting training with {len(train_examples)} examples")

        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        # Define loss function based on training mode
        if self.use_asymmetric_loss:
            # Use MultipleNegativesRankingLoss for asymmetric tasks (e.g., term-definition)
            # This loss function is ideal for information retrieval tasks where
            # queries (terms) need to match documents (definitions)
            train_loss = losses.MultipleNegativesRankingLoss(self.model)
            logger.info("Using MultipleNegativesRankingLoss for asymmetric task")
        else:
            # Use CosineSimilarityLoss for symmetric similarity tasks
            train_loss = losses.CosineSimilarityLoss(self.model)
            logger.info("Using CosineSimilarityLoss for symmetric task")

        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
        )

        logger.info(f"Training complete. Model saved to {output_path}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of embeddings
        """
        return self.model.encode(texts, convert_to_tensor=True)


def main():
    """Example usage of the EmbeddingFineTuner."""
    # Example training data: (text1, text2, similarity_score)
    sample_data = [
        ("This is a sample sentence", "This is another sentence", 0.8),
        ("Machine learning is fascinating", "Deep learning is interesting", 0.7),
        ("Python programming", "Java programming", 0.6),
        ("The weather is nice", "It's a beautiful day", 0.9),
        ("Embeddings are useful", "Vector representations are helpful", 0.85),
    ]

    # Initialize fine-tuner
    fine_tuner = EmbeddingFineTuner(model_name="BAAI/bge-small-en-v1.5")

    # Prepare training data
    train_examples = fine_tuner.prepare_training_data(sample_data)

    # Train the model
    fine_tuner.train(
        train_examples=train_examples,
        batch_size=2,
        epochs=1,
        output_path="./models/sample_model",
    )

    # Test encoding
    test_texts = ["This is a test sentence", "Another test sentence"]
    embeddings = fine_tuner.encode(test_texts)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
