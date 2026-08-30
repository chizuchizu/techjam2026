# tinyprof

An operator-level profiler for transformer inference on the ESP32-C3, built to
answer one question with evidence rather than arithmetic: **where did the 42.15 s
go, and what did the optimisation work actually buy?**

One capture produces one JSON artifact holding per-op time, call counts, derived
memory traffic, ELF-derived static memory, runtime heap and stack watermarks,
measured instrumentation overhead, and the accuracy gate. A second capture of a
different firmware revision renders against it as a single self-contained HTML
report.

It is deliberately small: two C files on the device, six Python files on the
host, no dependency beyond what this repo already uses (`numpy`, `pyserial`).

## Why it exists

The case-2 optimisation log is a careful record, but its measurements were made
by hand:

- per-op timings were transcribed from serial output into markdown;
- **every memory figure was copied out of PlatformIO's build summary** — nothing
  measured RAM at runtime;
- the "768 KB of flash reads per forward" figure was hand arithmetic, never
  checked against a measured call count;
- `device_test.py` printed results and persisted nothing, so no artifact linked a
  physical run to `score.py`'s MFU output.

Each of those is a place where a number can quietly stop being true. tinyprof
derives them instead, and cross-checks the derivations against each other.

## What it measures

| | source | |
|---|---|---|
| per-op time, exclusive and inclusive | 15 zones in `model.c` | measured |
| call counts | same | measured |
| instrumentation overhead | timed loop at dump time | measured |
| accuracy gate | per-element, against the torch reference | measured |
| heap free / min-ever / largest block | `esp_get_*_heap_size`, `heap_caps_*` | measured |
| stack high-water mark | `uxTaskGetStackHighWaterMark` | measured |
| static DRAM / IRAM / flash | `riscv32-esp-elf-size` on the linked ELF | measured |
| DRAM capacity | `dram0_0_seg` in the link map | measured |
| embedded weight blob sizes | `_binary_*_start/_end` symbols | measured |
| declared model workspace | arena census compiled into flash | measured |
| memory traffic | declared bytes/call × **measured** call count | modelled, derivation shown |
| MFU / roofline | imported from the project's own `tools/score.py` | derived |

Every derived value in the artifact carries `"measured": true` or a `"method"`
string, so `CONTRIBUTING.md`'s measured-versus-projected rule is enforced by the
schema rather than by remembering to write it down.

## Two measurement decisions worth knowing about

**Zones are timed with the cycle counter, not the microsecond timer.** The
implementation this replaced used `esp_timer_get_time()` and dropped any zone
that measured zero — so `res1` disappeared from the profile entirely and `res2`
reported 2 calls out of 12. A 6.25 ns tick (and counting every call, whether or
not it registered time) fixes a silent undercount in exactly the cheap ops an
optimisation pass is trying to drive toward zero. Zones shorter than 20 ticks are
flagged `resolution_limited` rather than reported as if exact.

**Nesting is declared by the firmware, not guessed by the host.** `quant` is
measured inside `qkv`, `gelu` inside `f2`, and the three attention phases inside
`attn`. Inclusive times would sum past 100% of the forward and a top-10 ranking
would double-count, so every ranking here uses exclusive time.

## Cost

Wiring tinyprof into the optimised firmware **frees 224 bytes of `.bss`** versus
the profiler it replaces (32-bit tick starts instead of 64-bit timestamps, and a
dump buffer moved to the stack), and leaves `.dram0.data` unchanged. That matters
because this build has very little DRAM left. Verify it after any change:

```sh
SZ=~/.platformio/packages/toolchain-riscv32-esp/bin/riscv32-esp-elf-size
for e in esp32-baseline esp32-tinyprof; do
  $SZ -A benchmarks/case-02/optimisation/esp32-baseline/.pio/build/$e/firmware.elf \
    | grep -E '\.dram0\.(bss|data)'
done
```

## Use

Both firmwares expose tinyprof behind their own PlatformIO env, so the benchmark
envs still compile the original code and cannot be moved by profiling work.

```sh
# whole pipeline against two boards
python3 tinyprof/tools/tinyprof.py case2 \
    --port /dev/ttyACM0 --port-v0 /dev/ttyACM1 --seeds 0 1 2 --reps 3

# one stage at a time
python3 tinyprof/tools/tinyprof.py collect --port /dev/ttyACM0 \
    --project benchmarks/case-02/optimisation/esp32-baseline \
    --env esp32-tinyprof --tag opt23 -o raw.json
python3 tinyprof/tools/tinyprof.py analyze raw.json \
    --project benchmarks/case-02/optimisation/esp32-baseline \
    --elf .../firmware.elf --map .../firmware.map -o artifact.json
```

### Without a board

The same `model.c` and `kernels.c` build natively with the same instrumentation,
so the whole pipeline runs on the host:

```sh
make -C tinyprof/tools host-profile host_profile_v0
python3 tinyprof/tools/tinyprof.py case2 --outdir /tmp/tp
```

Host artifacts are stamped `device: host` and the report opens with a banner
refusing to present them as ESP32 results — **and it means it**. On a host with a
hardware FPU the optimised build can be *slower* than the baseline, because the
optimisation trades float work for integer work that only pays off on a core with
no FPU. Structure, call counts, memory and traffic transfer to the device; times
do not.

## Layout

```
tinyprof/
  firmware/   tinyprof.h  tinyprof.c  tinyprof_esp32.c  library.json
  tools/      tinyprof.py         front end
              tp_collect.py       serial driver -> raw capture
              tp_parse.py         pure wire-format parser (no I/O)
              tp_analyze.py       raw -> artifact
              tp_elf.py           static memory from the ELF
              tp_compare.py       two artifacts -> comparison
              tp_report.py        comparison -> self-contained HTML
              tp_svg.py           dependency-free SVG primitives
              host_profile.c      native capture driver
              spec/case02.json    per-op traffic model, with derivations
```

## Out of scope

No instruction-level or PC sampling, no call-graph reconstruction, no automatic
instrumentation — zones are placed by hand. No multi-core support (the C3 has one
core). It profiles the forward pass, not the WiFi transport; that is
`esp32-linkbench`'s job.
