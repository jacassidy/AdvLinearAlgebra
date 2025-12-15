PYTHON := python3
PIP := $(PYTHON) -m pip

all: run

.PHONY: install run
install: requirements.txt
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) FunctionProjectionOntoPolynomails.py
