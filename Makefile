.PHONY: help install install-poetry run verify docker-build docker-run docker-compose-up clean

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies using pip"
	@echo "  make install-poetry   - Install dependencies using Poetry"
	@echo "  make run              - Run the training script"
	@echo "  make verify           - Verify the setup"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run training in Docker container"
	@echo "  make docker-compose-up - Run using Docker Compose"
	@echo "  make clean            - Clean up generated files"

install:
	pip install -r requirements.txt

install-poetry:
	poetry install

run:
	python src/main.py

verify:
	python verify_setup.py

docker-build:
	docker build -t embedding-finetuning .

docker-run:
	docker run -v $$(pwd)/finetuned_finance_model:/app/finetuned_finance_model embedding-finetuning

docker-compose-up:
	docker-compose up --build

clean:
	rm -rf finetuned_finance_model/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
