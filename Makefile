PYTHON ?= python3
CASE2_DIR := benchmarks/case-02
OPTIMISATION_DIR := $(CASE2_DIR)/optimisation
MULTIBOARD_DIR := $(CASE2_DIR)/multiboard
EXPERIMENTS_DIR := benchmarks/experiments
BASELINE_PROJECT := $(OPTIMISATION_DIR)/esp32-baseline
PYTHON_SOURCES := \
	torch_transformer_benchmark.py \
	$(wildcard $(BASELINE_PROJECT)/tools/*.py) \
	$(wildcard $(MULTIBOARD_DIR)/tools/*.py) \
	$(wildcard $(EXPERIMENTS_DIR)/*/tools/*.py) \
	$(wildcard archive/h200/*.py)

.PHONY: help check score host-test

help:
	@echo "make check      Python syntax and scoring smoke test"
	@echo "make score      Reproduce the ESP32 baseline score"
	@echo "make host-test  Run the baseline C accuracy suite (artifacts required)"

check:
	$(PYTHON) -m py_compile $(PYTHON_SOURCES)
	$(PYTHON) $(BASELINE_PROJECT)/tools/score.py \
		--runs $(BASELINE_PROJECT)/tools/runs.json --output /dev/null >/dev/null

score:
	$(PYTHON) $(BASELINE_PROJECT)/tools/score.py \
		--runs $(BASELINE_PROJECT)/tools/runs.json \
		--output $(BASELINE_PROJECT)/scores.json

host-test:
	$(MAKE) -C $(BASELINE_PROJECT)/tools test
