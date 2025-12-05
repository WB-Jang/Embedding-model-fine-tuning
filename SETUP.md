# 설정 가이드 (Setup Guide)

이 프로젝트를 실행하기 위한 상세한 설정 가이드입니다.

## 시스템 요구사항

- **Python**: 3.9 이상
- **GPU**: CUDA 호환 GPU 권장 (RTX 4060 이상, 8GB+ VRAM)
- **메모리**: 16GB RAM 권장
- **디스크 공간**: 최소 10GB (모델 다운로드 및 저장용)

## 설치 방법

### 방법 1: pip + 가상환경 (가장 간단)

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 학습 실행
python src/main.py
```

### 방법 2: Poetry (개발 환경 권장)

```bash
# 1. Poetry 설치 (이미 설치되어 있다면 생략)
curl -sSL https://install.python-poetry.org | python3 -

# 2. 의존성 설치
poetry install

# 3. 학습 실행
poetry run python src/main.py

# 또는 Poetry shell에 진입 후 실행
poetry shell
python src/main.py
```

### 방법 3: Docker (환경 독립성 보장)

```bash
# 1. Docker 이미지 빌드
docker build -t embedding-finetuning .

# 2. 컨테이너 실행
docker run -v $(pwd)/finetuned_finance_model:/app/finetuned_finance_model embedding-finetuning

# 또는 Docker Compose 사용
docker-compose up --build
```

#### GPU 지원 Docker 설정

GPU를 사용하려면:

1. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 설치
2. `docker-compose.yml` 파일에서 GPU 섹션의 주석 제거
3. 실행:

```bash
docker-compose up --build
```

### 방법 4: VS Code Dev Container (가장 편리)

1. VS Code에서 [Dev Containers 확장](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 설치
2. 프로젝트 폴더를 VS Code에서 열기
3. `F1` 키를 누르고 "Dev Containers: Reopen in Container" 선택
4. 컨테이너가 빌드되면 터미널에서 실행:

```bash
python src/main.py
```

## 빠른 시작 스크립트

자동 설정을 원하시면:

```bash
./quick_start.sh
```

이 스크립트는 대화형으로 설치 방법을 선택하고 자동으로 설정합니다.

## 설정 검증

설치가 완료되면 다음 명령으로 설정을 검증할 수 있습니다:

```bash
python verify_setup.py
```

## 학습 설정 커스터마이징

`src/main.py` 파일에서 다음 설정을 변경할 수 있습니다:

```python
MODEL_ID = "BAAI/bge-m3"              # 사용할 모델
OUTPUT_PATH = "./finetuned_finance_model"  # 모델 저장 경로
BATCH_SIZE = 4                        # 배치 크기 (메모리에 따라 조절)
NUM_EPOCHS = 3                        # 학습 반복 횟수
```

### 메모리 부족 (OOM) 해결

GPU 메모리가 부족하면 `BATCH_SIZE`를 줄이세요:

```python
BATCH_SIZE = 2  # 또는 1
```

## 데이터 준비

### 더미 데이터 사용 (기본값)

스크립트는 기본적으로 테스트용 더미 데이터를 사용합니다.

### 실제 데이터 사용

1. CSV 파일 준비 (컬럼: `term`, `definition`)
2. `src/main.py`의 데이터 로딩 부분 수정:

```python
# 이 부분의 주석을 해제하고 파일 경로 지정
df = pd.read_csv("your_data.csv")
```

예시 CSV 형식:

```csv
term,definition
RP,금융기관이 일정 기간 후에 다시 사는 조건으로 채권을 팔고...
GDP,국내총생산. 한 나라의 영역 내에서 가계 기업...
```

## 학습 실행

### 일반 실행

```bash
python src/main.py
```

### GPU 확인

학습 전에 GPU가 인식되는지 확인:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 학습 결과 사용

학습이 완료되면 `./finetuned_finance_model/` 디렉토리에 모델이 저장됩니다.

### 모델 로드 및 사용

```python
from sentence_transformers import SentenceTransformer

# 파인튜닝된 모델 로드
model = SentenceTransformer("./finetuned_finance_model")

# 텍스트 임베딩 생성
texts = ["GDP란 무엇인가?", "은행 건전성 지표는?"]
embeddings = model.encode(texts)

# 유사도 계산
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"유사도: {similarity}")
```

## Makefile 사용

프로젝트에는 편리한 Makefile이 포함되어 있습니다:

```bash
make help              # 사용 가능한 명령 보기
make install           # pip로 의존성 설치
make install-poetry    # Poetry로 의존성 설치
make run               # 학습 스크립트 실행
make verify            # 설정 검증
make docker-build      # Docker 이미지 빌드
make docker-run        # Docker로 실행
make docker-compose-up # Docker Compose로 실행
make clean             # 생성된 파일 정리
```

## 문제 해결

### CUDA 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결책**: `BATCH_SIZE`를 줄이세요 (예: 2 또는 1)

### 모델 다운로드 실패

```
Connection error downloading model
```

**해결책**: 
- 인터넷 연결 확인
- Hugging Face에서 수동으로 모델 다운로드
- 프록시 설정 확인

### 의존성 충돌

```
ERROR: Cannot install package conflicts
```

**해결책**:
```bash
# 가상환경을 새로 만들고 다시 설치
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 추가 리소스

- [Sentence Transformers 문서](https://www.sbert.net/)
- [BAAI/bge-m3 모델](https://huggingface.co/BAAI/bge-m3)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)

## 지원

문제가 발생하면 GitHub Issues에 보고해주세요.
