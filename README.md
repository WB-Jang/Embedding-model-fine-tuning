# Embedding Model Fine-tuning (Docker Environment)

임베딩 모델(bge-m3 등)을 파인튜닝하는 프로젝트입니다. RTX 4060 GPU를 사용하여 Jupyter Notebook 환경에서 학습할 수 있도록 Docker 컨테이너 환경으로 구성되어 있습니다.

## 📋 시스템 요구사항

- **Docker**: 20.10 이상
- **Docker Compose**: v2 이상 (또는 docker-compose v1.29+)
- **NVIDIA Docker Runtime**: GPU 사용을 위해 필요
- **GPU**: NVIDIA RTX 4060 (8GB VRAM) 또는 CUDA 12.x 호환 GPU
- **RAM**: 최소 16GB 권장
- **디스크 여유 공간**: 최소 30GB

## 🚀 빠른 시작

### 자동 설치 (권장)

```bash
chmod +x quick_start.sh
./quick_start.sh
```

대화형 메뉴에서 원하는 옵션을 선택하세요.

### 수동 설치

#### 1. Docker 컨테이너 빌드 및 실행

```bash
# Docker Compose로 빌드 및 실행
docker-compose up -d --build

# Jupyter Lab 접속
# 브라우저에서 http://localhost:8888 열기
```

#### 2. VSCode Dev Container 사용

1. VSCode에서 "Dev Containers" 확장 설치
2. 프로젝트 폴더 열기
3. `F1` → "Dev Containers: Reopen in Container" 선택
4. 컨테이너 내부에서 Jupyter Lab 실행:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### 3. GPU 지원 확인

컨테이너 내부에서:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## 📁 프로젝트 구조

```
Embedding-model-fine-tuning/
├── .devcontainer/
│   └── devcontainer.json          # VSCode Dev Container 설정
├── data/                          # 학습 데이터 (Excel 파일)
│   ├── 20251205_시사경제용어사전.xlsx
│   └── 한국은행_경제금융용어700선_정리.xlsx
├── src/
│   ├── main.py                    # 기본 학습 스크립트
│   ├── main_LoRA.py              # LoRA 파인튜닝 스크립트
│   └── api.py                     # API 서버
├── finetuned_finance_model/       # 학습된 모델 저장 위치
├── Embedding_model_fine_tuning_test.ipynb  # 메인 노트북
├── Dockerfile                     # Docker 이미지 정의
├── docker-compose.yml             # Docker Compose 설정
├── requirements.txt               # Python 의존성
├── DOCKER_SETUP.md               # 상세 Docker 설정 가이드
└── README.md                      # 이 파일
```

## 📓 Jupyter Notebook 실행

### 메인 노트북: `Embedding_model_fine_tuning_test.ipynb`

이 노트북은 다음 작업을 수행합니다:

1. **데이터 로드**: Excel 파일에서 경제/금융 용어 데이터 로드
2. **모델 다운로드**: Hugging Face에서 bge-m3 모델 다운로드
3. **LoRA 설정**: 효율적인 파인튜닝을 위한 LoRA 구성
4. **학습 실행**: MultipleNegativesRankingLoss로 모델 학습
5. **모델 저장**: 학습된 모델 저장 및 테스트

### 노트북 접속 방법

1. Docker 컨테이너 실행 후
2. 브라우저에서 `http://localhost:8888` 접속
3. `Embedding_model_fine_tuning_test.ipynb` 열기
4. 셀을 순서대로 실행

## 🔧 주요 설정

### 학습 파라미터 (노트북 내부)

- **Base Model**: BAAI/bge-m3 (다국어 임베딩 모델)
- **LoRA Config**:
  - rank (r): 8
  - alpha: 16
  - dropout: 0.1
- **Training**:
  - Batch size: 8-16 (GPU 메모리에 따라 조절)
  - Epochs: 3-5
  - Learning rate: 2e-5

### Docker 설정

#### GPU 메모리 조절

`docker-compose.yml`에서 shared memory 크기 조절:
```yaml
shm_size: '8gb'  # 필요시 증가
```

#### 포트 변경

8888 포트가 사용 중인 경우:
```yaml
ports:
  - "8889:8888"  # 호스트 포트 변경
```

## 📊 데이터 준비

학습 데이터는 `data/` 폴더에 위치:

- `data/20251205_시사경제용어사전.xlsx`
- `data/한국은행_경제금융용어700선_정리.xlsx`

### 데이터 형식

Excel 파일 구조:
- **용어** 열: 금융/경제 용어
- **설명** 열: 용어에 대한 정의

## 🎯 학습 프로세스

1. **데이터 전처리**: Excel → Pandas DataFrame
2. **학습 데이터 생성**: (용어, 정의) 쌍 생성
3. **모델 로드**: bge-m3 베이스 모델
4. **LoRA 적용**: 효율적인 파인튜닝
5. **학습**: MultipleNegativesRankingLoss
6. **저장**: `finetuned_finance_model/` 디렉토리

## 🔍 모델 사용

학습된 모델 로드:

```python
from sentence_transformers import SentenceTransformer

# 학습된 모델 로드
model = SentenceTransformer("./finetuned_finance_model")

# 텍스트 임베딩
query = "금융 파생상품이란?"
embeddings = model.encode([query])

# 유사도 검색
corpus_embeddings = model.encode(corpus_texts)
similarities = model.similarity(embeddings, corpus_embeddings)
```

## ⚠️ 문제 해결

### GPU가 인식되지 않는 경우

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Docker 재시작
sudo systemctl restart docker
```

### 메모리 부족 (OOM)

1. 배치 크기 줄이기 (노트북 내부)
2. `docker-compose.yml`에서 `shm_size` 증가
3. GPU 메모리 정리:
```python
import torch
torch.cuda.empty_cache()
```

### 컨테이너 로그 확인

```bash
# 실시간 로그
docker-compose logs -f

# 특정 컨테이너 로그
docker-compose logs embedding-finetuning
```

### 컨테이너 재시작

```bash
# 중지 및 재시작
docker-compose down
docker-compose up -d --build
```

## 📚 추가 문서

- **[DOCKER_SETUP.md](DOCKER_SETUP.md)**: 상세한 Docker 설정 가이드
- **[.docker-build-validation.md](.docker-build-validation.md)**: Docker 빌드 검증 결과

## 🛠️ 개발 환경

- Python: 3.11
- PyTorch: 2.5.1
- CUDA: 12.4
- cuDNN: 9
- sentence-transformers: >=2.2.0
- transformers: >=4.38.0
- peft: >=0.8.0

## 📝 참고사항

- 최초 빌드 시 PyTorch 이미지 다운로드로 인해 시간이 소요됩니다 (~15-20분)
- 학습된 모델은 호스트의 `finetuned_finance_model/` 디렉토리에 저장됩니다
- 컨테이너를 삭제해도 모델 파일은 유지됩니다 (볼륨 마운트)

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용 가능합니다.
