# tinyprof host-side build.
#
# host-profile builds the case-2 forward natively with tinyprof enabled, so the
# full collect -> analyze -> compare -> report pipeline can be exercised without
# a board. Artifacts it produces are stamped device=host and are never presented
# as ESP32 measurements.
#
#   make host-profile PROJECT=../../benchmarks/case-02/optimisation/esp32-baseline
#   make smoke                    # fixture-only pipeline check, no weights needed

CC      ?= cc
TP      := ../firmware
PROJECT ?= ../../benchmarks/case-02/optimisation/esp32-baseline
TAG     ?= opt23
CFLAGS  ?= -O2 -Wall -Wextra
LDLIBS  ?= -lm

SRCS = host_profile.c $(PROJECT)/src/kernels.c $(PROJECT)/src/model.c $(TP)/tinyprof.c

host-profile: host_profile

host_profile: $(SRCS) $(TP)/tinyprof.h
	$(CC) $(CFLAGS) -I$(PROJECT)/src -I$(TP) \
		-DTINYPROF_LIB=1 -DTINYPROF_TAG='"$(TAG)"' \
		-o $@ $(SRCS) $(LDLIBS)

# One host capture, straight to a raw artifact.
capture: host_profile
	./host_profile --root $(PROJECT) --seed 0 --reps 3 --fast

clean:
	rm -f host_profile

.PHONY: host-profile capture clean smoke

# --- restored pre-optimisation baseline -------------------------------------
# Same driver, same profiler, different firmware revision: that is the point.
# ATTN_PHASES defaults off for the host build because this firmware's attention
# is one fused j loop, so a phase split costs 132,096 zones per forward - noise
# against 42 s on the C3, larger than the work itself against 89 ms on a host.
V0      ?= ../../benchmarks/case-02/baseline/esp32-baseline-v0
V0TAG   ?= baseline-79f284a
PHASES  ?= 0

V0_SRCS = host_profile.c $(V0)/src/kernels.c $(V0)/src/model.c $(TP)/tinyprof.c

host_profile_v0: $(V0_SRCS) $(TP)/tinyprof.h
	$(CC) $(CFLAGS) -I$(V0)/src -I$(TP) \
		-DTINYPROF_LIB=1 -DTINYPROF_TAG='"$(V0TAG)"' \
		-DTINYPROF_ATTN_PHASES=$(PHASES) \
		-o $@ $(V0_SRCS) $(LDLIBS)
