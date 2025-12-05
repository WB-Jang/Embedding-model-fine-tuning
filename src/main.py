"""Main script for fine-tuning embedding models on term-definition pairs.

This script is designed for training on dictionary data like the
기획재정부 시사경제용어사전 (Ministry of Economy and Finance Current Economic Terms Dictionary).

The training data format is (term, definition) pairs, which represents
an asymmetric semantic search task where terms (queries) need to find
their corresponding definitions (documents).
"""

import json
import logging
from typing import List, Tuple
from embedding_finetune.train import EmbeddingFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dictionary_data(file_path: str) -> List[Tuple[str, str]]:
    """Load dictionary data from JSON file.

    Expected format:
    [
        {"term": "...", "definition": "..."},
        ...
    ]
    or
    {
        "term1": "definition1",
        "term2": "definition2",
        ...
    }

    Args:
        file_path: Path to JSON file containing term-definition pairs

    Returns:
        List of (term, definition) tuples
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    if isinstance(data, list):
        # Format: [{"term": "...", "definition": "..."}, ...]
        for item in data:
            pairs.append((item["term"], item["definition"]))
    elif isinstance(data, dict):
        # Format: {"term1": "definition1", ...}
        for term, definition in data.items():
            pairs.append((term, definition))
    else:
        raise ValueError("Unsupported data format")

    logger.info(f"Loaded {len(pairs)} term-definition pairs from {file_path}")
    return pairs


def main():
    """Main training script for term-definition embedding model."""
    # Example dictionary data: (term, definition)
    # This simulates data from 기획재정부 시사경제용어사전
    sample_dictionary_data = [
        (
            "GDP",
            "국내총생산(Gross Domestic Product)은 일정 기간 동안 한 나라 안에서 생산된 모든 최종 재화와 서비스의 시장 가치를 합한 것이다."
        ),
        (
            "인플레이션",
            "물가가 지속적으로 상승하는 경제 현상으로, 화폐 가치가 하락하고 구매력이 감소하는 것을 의미한다."
        ),
        (
            "양적완화",
            "중앙은행이 국채나 회사채 등을 매입하여 시중에 유동성을 공급함으로써 경기를 부양하는 통화정책이다."
        ),
        (
            "기준금리",
            "중앙은행이 금융기관에 자금을 대출할 때 적용하는 기준이 되는 금리로, 시중금리 수준을 조절하는 주요 정책수단이다."
        ),
        (
            "경상수지",
            "일정 기간 동안 한 나라가 외국과 거래한 재화·서비스 및 소득의 수입과 지출 차액을 나타내는 지표이다."
        ),
        (
            "재정정책",
            "정부가 조세 수입과 재정 지출의 규모를 조절하여 경제 안정과 성장을 도모하는 경제정책이다."
        ),
        (
            "환율",
            "자국 화폐와 외국 화폐의 교환 비율로, 외환시장에서의 수요와 공급에 따라 결정된다."
        ),
        (
            "디플레이션",
            "물가가 지속적으로 하락하는 경제 현상으로, 소비와 투자가 위축되어 경기 침체를 초래할 수 있다."
        ),
    ]

    # Initialize fine-tuner with appropriate model
    # For Korean language, consider using multilingual models or Korean-specific models
    logger.info("Initializing embedding model for term-definition training...")
    fine_tuner = EmbeddingFineTuner(
        model_name="BAAI/bge-m3",  # Multilingual model that supports Korean
        use_asymmetric_loss=True,  # Use asymmetric loss for term-definition pairs
    )

    # Prepare training data
    # Note: For term-definition pairs, we don't need similarity scores
    # The model will learn to match terms to their definitions
    train_examples = fine_tuner.prepare_term_definition_data(sample_dictionary_data)

    # Train the model
    logger.info("Starting training...")
    fine_tuner.train(
        train_examples=train_examples,
        batch_size=8,  # Adjust based on available GPU memory
        epochs=10,  # Increase epochs for better convergence on dictionary data
        output_path="./models/economic_terms_model",
        warmup_steps=100,
    )

    # Test the model
    logger.info("Testing the trained model...")
    test_terms = [
        "GDP",
        "인플레이션",
        "기준금리",
    ]

    # Encode terms (queries)
    term_embeddings = fine_tuner.encode(test_terms)
    logger.info(f"Term embeddings shape: {term_embeddings.shape}")

    # For actual use, you would encode all definitions and use similarity search
    # to find the most relevant definition for a given term
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
