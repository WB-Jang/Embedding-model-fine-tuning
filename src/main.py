import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# ---------------------------------------------------------
# 1. 설정 (Configuration)
# ---------------------------------------------------------
# RTX 4060 8GB를 고려한 설정입니다. OOM(메모리 부족) 발생 시 batch_size를 줄이세요.
MODEL_ID = "BAAI/bge-m3"  
OUTPUT_PATH = "./finetuned_finance_model"
BATCH_SIZE = 4            # BGE-M3가 크기 때문에 4~8 권장 (8GB VRAM 기준)
NUM_EPOCHS = 3            # 데이터 양에 따라 조절 (보통 1~3회면 충분)

# ---------------------------------------------------------
# 2. 데이터 준비 (Data Preparation)
# ---------------------------------------------------------
# 가정: 기재부 용어사전이 CSV 파일로 있고, 컬럼명이 'term'(용어), 'definition'(설명)이라고 가정
# 실제 데이터가 없다면 아래 dummy_data를 사용해 테스트해보세요.

# [실제 사용 시 CSV 로드]
# df = pd.read_csv("moef_dictionary.csv") 

# [테스트용 더미 데이터 생성]
data = [
    {"term": "RP", "definition": "금융기관이 일정 기간 후에 다시 사는 조건으로 채권을 팔고 경과 기간에 따라 소정의 이자를 붙여 되사는 채권."},
    {"term": "GDP", "definition": "국내총생산. 한 나라의 영역 내에서 가계, 기업, 정부 등 모든 경제주체가 일정기간 동안 생산한 재화 및 서비스의 부가가치를 시장가격으로 평가하여 합산한 것."},
    {"term": "BIS 자기자본비율", "definition": "국제결제은행이 정한 은행의 위험자산 대비 자기자본비율로, 은행 건전성을 나타내는 핵심 지표."},
    # ... 실제 데이터를 여기에 추가하세요
]
df = pd.DataFrame(data)

train_examples = []

print(f"총 {len(df)}개의 용어 데이터를 학습 데이터로 변환합니다.")

for i, row in df.iterrows():
    term = row['term']
    definition = row['definition']
    
    # [중요] 데이터 증강 (Data Augmentation)
    # 사용자가 검색할 때 단순히 용어만 칠 수도 있지만, 질문 형태로 칠 수도 있습니다.
    # 모델이 다양한 질문 패턴을 익히도록 '질문(Query)'을 다양하게 만들어줍니다.
    
    queries = [
        f"{term}",                 # 단순히 용어만 검색
        f"{term}의 뜻은?",          # 질문 형태
        f"{term}이란 무엇인가?",    # 질문 형태 2
        f"금융 용어 {term} 설명"    # 키워드 조합
    ]
    
    # 하나의 설명(Passage)에 대해 여러 개의 질문(Query)을 매칭하여 학습 데이터 생성
    for query in queries:
        train_examples.append(InputExample(texts=[query, definition]))

# ---------------------------------------------------------
# 3. 모델 및 로더 초기화 (Load Model & DataLoader)
# ---------------------------------------------------------
print(f"모델 로드 중: {MODEL_ID} ...")
model = SentenceTransformer(MODEL_ID)

# 데이터 로더 설정
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# ---------------------------------------------------------
# 4. 손실함수 설정 (Loss Function)
# ---------------------------------------------------------
# MultipleNegativesRankingLoss: 
# (질문, 정답) 쌍을 가까워지게 하고, 배치 내의 다른 (질문, 오답) 쌍은 멀어지게 학습합니다.
# 검색(Retrieval) 모델 학습에 가장 표준적으로 사용되는 함수입니다.
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# ---------------------------------------------------------
# 5. 학습 시작 (Training)
# ---------------------------------------------------------
print("학습을 시작합니다... (GPU 상태에 따라 시간이 소요됩니다)")

# Warmup step은 전체 학습 단계의 10% 정도로 설정
warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

print(f"\n학습 완료! 모델이 '{OUTPUT_PATH}'에 저장되었습니다.")

# ---------------------------------------------------------
# 6. 간단 검증 (Validation)
# ---------------------------------------------------------
print("\n[성능 테스트]")
finetuned_model = SentenceTransformer(OUTPUT_PATH)

test_query = "은행의 건전성을 보여주는 지표는?"
# 학습 데이터에 있던 'BIS 자기자본비율' 설명과 매칭되는지 확인
docs = [
    "국내총생산은 한 나라의 경제 규모를 나타낸다.", 
    "국제결제은행이 정한 위험자산 대비 자기자본비율로 은행 건전성 핵심 지표이다.", # 정답
    "RP는 환매조건부채권을 의미한다."
]

embeddings_query = finetuned_model.encode(test_query)
embeddings_docs = finetuned_model.encode(docs)

# 유사도 계산
similarities = finetuned_model.similarity(embeddings_query, embeddings_docs)
print(f"질문: {test_query}")
for i, doc in enumerate(docs):
    print(f"문서 {i+1} 유사도: {similarities[0][i]:.4f} -> {doc}")
