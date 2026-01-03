# Docker Setup Guide for Embedding Model Fine-tuning

이 문서는 RTX 4060 GPU를 사용하여 `Embedding_model_fine_tuning_test.ipynb` 노트북을 실행할 수 있는 Docker 환경 설정 방법을 설명합니다.

## 사전 요구사항

### 1. Docker 및 NVIDIA Container Toolkit 설치

#### Docker 설치
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### NVIDIA Container Toolkit 설치 (GPU 지원)
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### GPU 확인
```bash
nvidia-smi
```

### 2. Docker Compose 설치
```bash
sudo apt-get install docker-compose-plugin
```

## 실행 방법

### 방법 1: Docker Compose 사용 (권장)

1. **컨테이너 빌드 및 시작**
```bash
docker-compose up -d --build
```

2. **Jupyter Lab 접속**
브라우저에서 다음 주소로 접속:
```
http://localhost:8888
```

3. **컨테이너 로그 확인**
```bash
docker-compose logs -f
```

4. **컨테이너 중지**
```bash
docker-compose down
```

### 방법 2: VSCode Dev Container 사용

1. **VSCode 확장 설치**
   - "Dev Containers" 확장 설치

2. **컨테이너에서 열기**
   - `F1` 또는 `Ctrl+Shift+P` 눌러 명령 팔레트 열기
   - "Dev Containers: Reopen in Container" 선택

3. **Jupyter 실행**
   컨테이너 내부 터미널에서:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
   ```

### 방법 3: Docker 직접 사용

1. **이미지 빌드**
```bash
docker build -t embedding-finetuning:latest .
```

2. **컨테이너 실행**
```bash
docker run -it --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/finetuned_finance_model:/app/finetuned_finance_model \
  -v $(pwd)/Embedding_model_fine_tuning_test.ipynb:/app/Embedding_model_fine_tuning_test.ipynb \
  embedding-finetuning:latest
```

## 디렉토리 구조

```
Embedding-model-fine-tuning/
├── .devcontainer/
│   └── devcontainer.json          # VSCode Dev Container 설정
├── data/                          # 학습 데이터 폴더
│   ├── 20251205_시사경제용어사전.xlsx
│   └── 한국은행_경제금융용어700선_정리.xlsx
├── src/                           # 소스 코드
│   ├── main.py
│   ├── main_LoRA.py
│   └── api.py
├── finetuned_finance_model/       # 학습된 모델 저장 (자동 생성)
├── Embedding_model_fine_tuning_test.ipynb  # 메인 노트북
├── Dockerfile                     # Docker 이미지 정의
├── docker-compose.yml             # Docker Compose 설정
├── requirements.txt               # Python 의존성
└── README.md
```

## GPU 지원 확인

컨테이너 내부에서 GPU가 인식되는지 확인:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

예상 출력:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4060
```

## 주요 라이브러리 버전

- **PyTorch**: 2.5.1
- **CUDA**: 12.4
- **cuDNN**: 9
- **Python**: 3.11 (PyTorch 이미지 기본값)
- **sentence-transformers**: >=2.2.0
- **transformers**: >=4.38.0
- **peft**: >=0.8.0

## 문제 해결

### GPU가 인식되지 않는 경우

1. NVIDIA 드라이버 확인:
```bash
nvidia-smi
```

2. Docker가 GPU를 사용할 수 있는지 확인:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

3. Docker Compose에서 GPU가 활성화되었는지 확인:
```bash
docker-compose config
```

### 메모리 부족 오류

`docker-compose.yml`에서 `shm_size`를 늘리거나 배치 크기를 줄이세요:

```yaml
shm_size: '16gb'  # 기본값은 8gb
```

### 포트 충돌

8888 포트가 이미 사용 중인 경우, `docker-compose.yml`에서 포트를 변경:

```yaml
ports:
  - "8889:8888"  # 호스트 포트를 8889로 변경
```

## 데이터 준비

학습 데이터는 `data/` 폴더에 위치해야 합니다:

- `data/20251205_시사경제용어사전.xlsx`
- `data/한국은행_경제금융용어700선_정리.xlsx`

## 학습된 모델 저장

학습된 모델은 `finetuned_finance_model/` 디렉토리에 자동으로 저장됩니다.
이 디렉토리는 호스트와 컨테이너 간에 마운트되어 있어 컨테이너를 종료해도 모델이 유지됩니다.

## 추가 정보

- Docker 이미지 크기: 약 15-20GB (CUDA 및 PyTorch 포함)
- 권장 시스템 요구사항:
  - GPU: NVIDIA RTX 4060 (8GB VRAM)
  - RAM: 최소 16GB
  - 디스크 여유 공간: 최소 30GB

## 참고 자료

- [PyTorch Docker 이미지](https://hub.docker.com/r/pytorch/pytorch)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
