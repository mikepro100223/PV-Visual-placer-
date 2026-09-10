.PHONY: setup test coverage app doctor data train evaluate

setup:
	uv sync --frozen --python 3.12

doctor:
	uv run rooftop-pv doctor

test:
	uv run ruff check code tests scripts
	uv run pytest -q

coverage:
	uv run pytest --cov=code --cov-report=term-missing:skip-covered --cov-fail-under=80 -q

app:
	uv run streamlit run code/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

data:
	uv run rooftop-pv download-data
	uv run rooftop-pv prepare-data

train:
	uv run rooftop-pv train --config configs/train.yaml

evaluate:
	uv run rooftop-pv evaluate --split test
