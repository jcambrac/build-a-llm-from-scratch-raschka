VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install run activate

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) main.py

# Make cannot activate a venv in your current shell.
# Each recipe runs in a subshell that exits when make finishes.
# `source` is also a bash builtin, so /bin/sh reports "No such file or directory".
activate:
	@echo "Run this in your shell instead:"
	@echo "  source $(VENV)/bin/activate"
