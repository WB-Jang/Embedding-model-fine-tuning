from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
MODEL_PATH = "./finetuned_finance_model"

app = FastAPI(
    title="Embedding Fine-tuning API",
    description="Fine-tuned SentenceTransformer 모델로부터 임베딩을 제공하는 API",
    version="0.1.0",
)

# 애플리케이션 시작 시 한 번만 모델 로드
# (컨테이너가 GPU에 접근 가능한 상태여야 함)
@app.on_event("startup")
def load_model() -> None:
    global model
    model = SentenceTransformer(MODEL_PATH)
    # 필요하다면 device 강제 설정도 가능 (예: 'cuda' 또는 'cpu')
    # model = SentenceTransformer(MODEL_PATH, device="cuda")


# ---------------------------------------------------------
# 요청/응답 스키마
# ---------------------------------------------------------
class EncodeRequest(BaseModel):
    text: str


class BatchEncodeRequest(BaseModel):
    texts: List[str]


class EncodeResponse(BaseModel):
    embedding: List[float]


class BatchEncodeResponse(BaseModel):
    embeddings: List[List[float]]


# ---------------------------------------------------------
# 엔드포인트 정의
# ---------------------------------------------------------
@app.get("/health")
def health_check() -> dict:
    """
    헬스 체크용 엔드포인트.
    """
    return {"status": "ok"}


@app.post("/encode", response_model=EncodeResponse)
def encode(request: EncodeRequest) -> EncodeResponse:
    """
    단일 텍스트에 대한 임베딩을 반환합니다.
    """
    embedding = model.encode(request.text)
    # 넘파이 배열을 JSON으로 직렬화하기 위해 파이썬 리스트로 변환
    return EncodeResponse(embedding=embedding.tolist())


@app.post("/encode_batch", response_model=BatchEncodeResponse)
def encode_batch(request: BatchEncodeRequest) -> BatchEncodeResponse:
    """
    여러 텍스트에 대한 임베딩을 한 번에 반환합니다.
    """
    embeddings = model.encode(request.texts)
    return BatchEncodeResponse(embeddings=[vec.tolist() for vec in embeddings])
