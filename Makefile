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
	$(wildcard tinyprof/tools/*.py) \
	$(wildcard docs/report/*.py) \
	$(wildcard archive/h200/*.py)

.PHONY: help check score host-test tinyprof

help:
	@echo "make check      Python syntax, scoring smoke test, and the tinyprof selftest"
	@echo "make score      Reproduce the ESP32 baseline score"
	@echo "make host-test  Run the baseline C accuracy suite (artifacts required)"
	@echo "make tinyprof   Replay the committed tinyprof fixtures (no board needed)"

check: tinyprof
	$(PYTHON) -m py_compile $(PYTHON_SOURCES)
	$(PYTHON) $(BASELINE_PROJECT)/tools/score.py \
		--runs $(BASELINE_PROJECT)/tools/runs.json --output /dev/null >/dev/null

# Replays two committed captures through parse -> analyze -> compare -> report
# and asserts the invariants. Needs no board, no weights and no build, so a
# regression in the traffic model or the zone-nesting table fails CI rather than
# surfacing later as a wrong number in a report.
tinyprof:
	cd tinyprof/tools && $(PYTHON) tp_selftest.py

score:
	$(PYTHON) $(BASELINE_PROJECT)/tools/score.py \
		--runs $(BASELINE_PROJECT)/tools/runs.json \
		--output $(BASELINE_PROJECT)/scores.json

host-test:
	$(MAKE) -C $(BASELINE_PROJECT)/tools test
