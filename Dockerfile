# PyTorch 2.2 + CUDA 12.1 + cuDNN 8이 포함된 공식 이미지 (GPU용)
# 최신 버전으로 업데이트 함
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
# (공식 권장 스크립트 사용 후, PATH에 추가)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry

# pyproject.toml 및 poetry.lock만 먼저 복사 (의존성 캐시를 위해)
COPY pyproject.toml poetry.lock* ./

# Poetry를 전역 환경에 설치하도록 설정하고, 의존성 설치
# (virtualenvs.create=false 이면 컨테이너의 전역 Python 환경에 설치됨)
# --no-update 옵션으로 기존 torch 및 CUDA 패키지를 보호
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root || \
    (echo "Retrying with pip fallback..." && \
     pip install --no-cache-dir sentence-transformers transformers accelerate datasets fastapi uvicorn[standard] pandas)


# 애플리케이션 코드 복사
COPY src/ ./src/

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 모델 결과 저장 디렉토리 생성
RUN mkdir -p /app/finetuned_finance_model

# 컨테이너 기본 실행 명령: main.py 실행
# CMD ["python", "src/main.py"]
