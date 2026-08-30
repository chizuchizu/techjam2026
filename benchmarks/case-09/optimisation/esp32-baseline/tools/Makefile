# Host-side validation of the ESP32 kernels (macOS/Linux).
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -I../src
LDLIBS  ?= -lm

SRCS = host_test.c ../src/kernels.c ../src/model.c
TILED_SRCS = host_test.c ../src/kernels.c ../src/model.c ../src/model_tiled.c

host_test: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)
	@echo "run: ./tools/host_test all --both --reps 5 (from esp32-baseline)"

host_test_tiled: $(TILED_SRCS)
	$(CC) $(CFLAGS) -DTM_TILED_FORWARD -DTM_TILE_ROWS=16 \
		-DTM_A16_ROWS=16 -o $@ $(TILED_SRCS) $(LDLIBS)
	@echo "run: ./tools/host_test_tiled all --fast (from esp32-baseline)"

# regenerate torch artifacts with the system python3
export:
	python3 export_case2.py --outdir .. --seeds 25

# run the full 25-seed gate for both modes
test: host_test
	./host_test all --both --reps 5

bench: host_test
	./host_test all --both --bench --reps 5

clean:
	rm -f host_test host_test_tiled

clean-generated: clean
	rm -f ../weights.bin ../weights_q12.bin ../manifest.json
	rm -rf ../testdata

.PHONY: export test bench clean clean-generated
