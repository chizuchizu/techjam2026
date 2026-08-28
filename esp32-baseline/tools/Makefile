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

clean:
	rm -f host_test ../weights.bin ../weights_q12.bin
	rm -rf ../testdata
.PHONY: export test bench clean
