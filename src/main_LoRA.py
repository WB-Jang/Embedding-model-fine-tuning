import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------
# 1. 설정 (Configuration)
# ---------------------------------------------------------
MODEL_ID = "BAAI/bge-m3"
OUTPUT_PATH = "./finetuned_finance_model_lora"
BATCH_SIZE = 4
NUM_EPOCHS = 3

# ---------------------------------------------------------
# 2. 데이터 준비 (Data Preparation)
# ---------------------------------------------------------
data = [
    {"term": "RP", "definition": "금융기관이 일정 기간 후에 다시 사는 조건으로 채권을 팔고 경과 기간에 따라 소정의 이자를 붙여 되사는 채권."},
    {"term": "GDP", "definition": "국내총생산. 한 나라의 영역 내에서 가계, 기업, 정부 등 모든 경제주체가 일정기간 동안 생산한 재화 및 서비스의 부가가치 합계."},
    {"term": "BIS 자기자본비율", "definition": "국제결제은행이 정한 은행의 위험자산 대비 자기자본비율로, 은행 건전성을 나타내는 핵심 지표."},
]
df = pd.DataFrame(data)

train_examples = []

print(f"총 {len(df)}개의 용어 데이터를 학습 데이터로 변환합니다.")

for i, row in df.iterrows():
    term = row['term']
    definition = row['definition']

    queries = [
        f"{term}",
        f"{term}의 뜻은?",
        f"{term}이란 무엇인가?",
        f"금융 용어 {term} 설명"
    ]

    for query in queries:
        train_examples.append(InputExample(texts=[query, definition]))

# ---------------------------------------------------------
# 3. SentenceTransformer + LoRA 설정
# ---------------------------------------------------------
print(f"모델 로드 중: {MODEL_ID} ...")
model = SentenceTransformer(MODEL_ID)

# SentenceTransformer 내부의 transformer backbone 가져오기
# (BGE 계열은 일반적으로 첫 번째 모듈이 Transformer 입니다.)
base_module = model._first_module()  # sentence_transformers.models.Transformer
hf_model = base_module.auto_model     # transformers 모델 (e.g., BertModel/RoFormer 등)

# LoRA 설정
lora_config = LoraConfig(
    r=8,                # 랭크 (작을수록 가벼움)
    lora_alpha=16,      # LoRA scaling
    target_modules=["query", "value", "key"],  # attention module에만 적용 (모델 구조에 따라 조정 필요)
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",  # 임베딩/인코더 용도
)

hf_model = get_peft_model(hf_model, lora_config)

# SentenceTransformer 내부 모델에 다시 세팅
base_module.auto_model = hf_model

# 기존 파라미터는 freeze, LoRA 파라미터만 학습
for name, param in hf_model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False

print("학습 가능한 파라미터 수:")
trainable_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in hf_model.parameters())
print(f"  trainable: {trainable_params:,} / total: {all_params:,}")

# ---------------------------------------------------------
# 4. 데이터 로더 & 손실 함수
# ---------------------------------------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

train_loss = losses.MultipleNegativesRankingLoss(model=model)

# ---------------------------------------------------------
# 5. 학습 시작 (Training)
# ---------------------------------------------------------
print("학습을 시작합니다... (LoRA 기반 경량 파인튜닝)")

warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

print(f"\n학습 완료! LoRA 파라미터가 포함된 모델이 '{OUTPUT_PATH}'에 저장되었습니다.")

# ---------------------------------------------------------
# 6. 간단 검증 (Validation)
# ---------------------------------------------------------
print("\n[성능 테스트]")
finetuned_model = SentenceTransformer(OUTPUT_PATH)

test_query = "은행의 건전성을 보여주는 지표는?"
docs = [
    "국내총생산은 한 나라의 경제 규모를 나타낸다.", 
    "국제결제은행이 정한 위험자산 대비 자기자본비율로 은행 건전성 핵심 지표이다.",  # 정답
    "RP는 환매조건부채권을 의미한다."
]

embeddings_query = finetuned_model.encode(test_query)
embeddings_docs = finetuned_model.encode(docs)

similarities = finetuned_model.similarity(embeddings_query, embeddings_docs)
print(f"질문: {test_query}")
for i, doc in enumerate(docs):
    print(f"문서 {i+1} 유사도: {similarities[0][i]:.4f} -> {doc}")
