# PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9 (RTX 4060 supports CUDA 12.x)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# requirements.txt 복사 및 의존성 설치 (캐시 활용을 위해 먼저 복사)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY src/ ./src/

# Jupyter notebook 복사 (선택적)
COPY Embedding_model_fine_tuning_test.ipynb ./

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 모델 결과 저장 디렉토리 생성
RUN mkdir -p /app/finetuned_finance_model

# Jupyter 설정 디렉토리 생성
RUN mkdir -p /root/.jupyter

# 기본 실행 명령 (Jupyter Lab)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
