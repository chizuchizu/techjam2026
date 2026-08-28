PYTHON ?= python3

.PHONY: help check score host-test

help:
	@echo "make check      Python syntax and scoring smoke test"
	@echo "make score      Reproduce the ESP32 baseline score"
	@echo "make host-test  Run the baseline C accuracy suite (artifacts required)"

check:
	$(PYTHON) -m py_compile tools/*.py esp32-baseline/tools/*.py
	$(PYTHON) esp32-baseline/tools/score.py \
		--runs esp32-baseline/tools/runs.json --output /dev/null >/dev/null

score:
	$(PYTHON) esp32-baseline/tools/score.py \
		--runs esp32-baseline/tools/runs.json \
		--output esp32-baseline/scores.json

host-test:
	$(MAKE) -C esp32-baseline/tools test
