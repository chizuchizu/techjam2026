# Host validation of the two-node sequence-sharded forward.
BASE    ?= ../../../optimisation/esp32-baseline
CC      ?= cc
# TM_A16_ROWS matches the firmware build so the host gate exercises the
# same buffer geometry as the boards.
CFLAGS  ?= -O2 -Wall -Wextra -DTM_A16_ROWS=64 -I../src -I$(BASE)/src
LDLIBS  ?= -lm

SRCS = shard_host_test.c ../src/model_shard.c $(BASE)/src/kernels.c

shard_host_test: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)

test: shard_host_test
	./shard_host_test all

clean:
	rm -f shard_host_test

.PHONY: test clean
