# host-side validation of the ESP32 kernels (macOS/linux, no toroid needed here)
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -Isrc
LDLIBS  ?= -lm

SRCS = tools/host_test.c src/kernels.c src/model.c

host_test: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)
	@echo "run: ./tools/host_test all --both --reps 5"

# regenerate torch artifacts with the system python3
export:
	python3 tools/export_case2.py --outdir . --seeds 25

# run the full 25-seed gate for both modes
test: host_test
	./tools/host_test all --both --reps 5

bench: host_test
	./tools/host_test all --both --bench --reps 5

# Deliberately narrower than the original: weights.bin, weights_q12.bin and
# testdata/ in this project are symlinks to the shared blobs under
# ../../optimisation/esp32-baseline/, so that both firmwares are measured
# against byte-identical inputs. Removing them here would break the other
# project, and regenerating them would break the comparison.
clean:
	rm -f host_test
.PHONY: export test bench clean
