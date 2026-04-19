.PHONY: install download eda preprocess train explain evaluate all clean lint dashboard dashboard-prod up down logs pipeline-run

PYTHON := PYTHONPATH=. .venv/bin/python
PIP := .venv/bin/pip
SCRIPTS := scripts

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) $(SCRIPTS)/01_download.py

eda:
	$(PYTHON) $(SCRIPTS)/01b_eda.py

preprocess:
	$(PYTHON) $(SCRIPTS)/02_preprocess.py

train:
	$(PYTHON) $(SCRIPTS)/03_train_baseline.py
	$(PYTHON) $(SCRIPTS)/04_train_rf.py
	$(PYTHON) $(SCRIPTS)/05_train_catboost.py
	$(PYTHON) $(SCRIPTS)/05b_train_lightgbm.py
	$(PYTHON) $(SCRIPTS)/05c_train_xgboost.py
	$(PYTHON) $(SCRIPTS)/06_calibrate.py

explain:
	$(PYTHON) $(SCRIPTS)/07_explain_shap.py

evaluate:
	$(PYTHON) $(SCRIPTS)/08_evaluate_all.py

all: download eda preprocess train explain evaluate

dashboard:
	$(PYTHON) -m app.main

dashboard-prod:
	PYTHONPATH=. .venv/bin/gunicorn app.main:server --bind 0.0.0.0:8050 --workers 2

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

pipeline-run:
	docker compose --profile tools run --rm pipeline make all

lint:
	$(PYTHON) -m ruff check src/ scripts/ || true

clean:
	rm -rf data/interim data/processed models/*.pkl models/*.cbm reports/figures
	rm -f reports/metrics.json reports/comparison_table.md reports/shap_top_features.md reports/eda_summary.md reports/training_log.txt
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
