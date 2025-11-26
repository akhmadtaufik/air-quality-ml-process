.PHONY: help install test train serve-api serve-ui clean lint format

CONDA_ENV = pacmann
PYTHON = python3

help:
	@echo "Available commands:"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make pipeline-all   - Run complete pipeline (preprocess->train->deploy)"
	@echo "  make preprocess     - Preprocess raw data"
	@echo "  make train          - Train models"
	@echo "  make deploy-model   - Deploy to production"
	@echo ""
	@echo "Server Commands:"
	@echo "  make serve-api      - Start FastAPI server"
	@echo "  make serve-ui       - Start Streamlit UI"
	@echo ""
	@echo "Development Commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code"
	@echo "  make clean          - Clean temporary files"

install:
	conda run -n $(CONDA_ENV) pip install -r requirements.txt

test:
	@echo "Running test suite..."
	conda run -n $(CONDA_ENV) $(PYTHON) src/utils/run_tests.py

test-quick:
	conda run -n $(CONDA_ENV) pytest tests/ -v

preprocess:
	@echo "Running preprocessing pipeline..."
	conda run -n $(CONDA_ENV) $(PYTHON) main.py preprocess

train:
	@echo "Running training pipeline..."
	conda run -n $(CONDA_ENV) $(PYTHON) main.py train

deploy-model:
	@echo "Deploying model to production..."
	conda run -n $(CONDA_ENV) $(PYTHON) main.py deploy

pipeline-all:
	@echo "Running complete pipeline..."
	conda run -n $(CONDA_ENV) $(PYTHON) main.py all

serve-api:
	@echo "Starting FastAPI server on http://localhost:8000"
	conda run -n $(CONDA_ENV) $(PYTHON) main.py serve

serve-ui:
	@echo "Starting Streamlit UI on http://localhost:8501"
	@echo "Note: Press Ctrl+C to stop"
	conda run -n $(CONDA_ENV) streamlit run src/serving/ui.py --server.headless true

lint:
	conda run -n $(CONDA_ENV) flake8 src/ tests/ --max-line-length=79 --ignore=E501,W503

format:
	conda run -n $(CONDA_ENV) black src/ tests/ --line-length=79

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cleaned temporary files"

check-env:
	@echo "Checking conda environment: $(CONDA_ENV)"
	@conda info --envs | grep $(CONDA_ENV) || (echo "Environment $(CONDA_ENV) not found!" && exit 1)
	@echo "Environment OK"

api-test:
	@echo "Testing API endpoint..."
	curl -X GET http://localhost:8000/health || echo "API is not running"

notebook:
	conda run -n $(CONDA_ENV) jupyter notebook

setup-dev:
	conda run -n $(CONDA_ENV) pip install black flake8 pytest pytest-cov jupyter

all: install test

status:
	@echo "Checking project status..."
	conda run -n $(CONDA_ENV) $(PYTHON) main.py status
