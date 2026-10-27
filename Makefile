# TooClose developer commands. Prefer Python 3.11 for MediaPipe compatibility.
PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
MODEL := models/hand_landmarker.task
MODEL_URL := https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

.PHONY: venv install model setup test train debug dev frontend server clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

model:
	mkdir -p models
	@if [ ! -f "$(MODEL)" ]; then \
		echo "Downloading MediaPipe hand landmarker (local file, no CDN at runtime)..."; \
		curl -L --fail -o "$(MODEL)" "$(MODEL_URL)"; \
	else \
		echo "Model already present: $(MODEL)"; \
	fi

install: venv model
	@if [ -f frontend/package.json ]; then npm --prefix frontend install; fi

setup: install

test:
	$(BIN)/pytest

train:
	$(BIN)/python -m tooclose.train

debug:
	$(BIN)/python -m tooclose.debug_view

server:
	$(BIN)/uvicorn tooclose.server:app --host 127.0.0.1 --port 8000 --reload --app-dir src

frontend:
	npm --prefix frontend run dev

# Start API + Vite together. Ctrl-C stops both.
dev:
	@trap 'kill 0' INT TERM EXIT; \
	$(BIN)/uvicorn tooclose.server:app --host 127.0.0.1 --port 8000 --reload --app-dir src & \
	npm --prefix frontend run dev & \
	wait

clean:
	rm -rf $(VENV) .pytest_cache src/tooclose.egg-info frontend/dist frontend/node_modules
