# 25 — tinyprof: measured profiling instead of transcribed profiling

Date: 2026-08-30 · Tool: [`tinyprof/`](../../../../../tinyprof/) · Firmwares:
`esp32-tinyprof` (opt23) and `esp32-tinyprof-v0` (restored `79f284a`)

This entry adds no optimisation. It replaces the *method* by which entries 00
and 20 were produced, because several of their numbers were hand-made and could
not be re-derived.

## What was hand-made before

| figure | in | how it was obtained |
|---|---|---|
| per-op ms/forward | `00`, `20` | read off serial output, typed into markdown |
| SRAM / flash usage | `00`, `20` | copied from PlatformIO's build summary |
| 768 KB flash reads/forward | `20` §2 | arithmetic on assumed call counts |
| DRAM capacity 320,784 B | `20` §1 | constant, not read from the build |

None of these were wrong when written. All of them were unverifiable afterwards.

## What tinyprof derives instead

- **Per-op time from a 6.25 ns tick counter, not the 1 µs timer.** The previous
  block guarded with `if (d > 0)`, so any zone that rounded to zero was dropped
  from *both* its total and its call count. Measured effect: `res1` was absent
  from the profile entirely and `res2` reported 2 calls out of 12.
- **Exclusive time.** `quant` sits inside `qkv`, `gelu` inside `f2`, and the
  three attention phases inside `attn`. The nesting is declared by the firmware
  and subtracted host-side; exclusive times now sum to 99.2% of the measured
  forward, which is the check that the tree is right.
- **Static memory from the ELF**, with the DRAM denominator parsed from
  `dram0_0_seg` in the link map: **321,296 B**, not the 320,784 B constant.
- **Flash traffic from measured call counts.** `quant` is 48 calls/forward ×
  8,192 B = 393,216 B; `oproj`/`f1`/`f2` are 4 × 32,768 B each. Total
  **786,432 B = 768 KiB**, reproducing `20` §2 from first principles — plus
  9,216 B of LayerNorm gain/bias reads that the hand calculation omitted.
- **A cross-check between two independent sources.** The firmware's arena census
  (253,952 B) against the ELF's `.dram0.bss` (267,032 B) leaves 13,080 B of
  framework static state. This check is what found the 32 KB `a16` scratch
  buffer missing from the census.

## Cost to the optimised build

Instrumenting with tinyprof **reduces** `.dram0.bss` by 224 B versus the block it
replaces (32-bit tick starts rather than 64-bit timestamps; the dump buffer moved
to the stack), with `.dram0.data` unchanged:

| section | `esp32-baseline` | `esp32-tinyprof` | delta |
|---|---:|---:|---:|
| `.dram0.bss` | 267,256 | 267,032 | **−224** |
| `.dram0.data` | 7,308 | 7,308 | 0 |
| `.flash.text` | 167,696 | 166,356 | −1,340 |
| `.flash.rodata` | 2,423,528 | 2,423,912 | +384 |

The benchmark envs are untouched: `TINYPROF_LIB` gates the whole thing, so
`esp32-baseline` still compiles the original inline profiler and the published
1.996 s/forward cannot be moved by profiling work.

## Restored baseline firmware

The 42.15 s starting point had no source on disk — `baseline/` held only a log.
[`baseline/esp32-baseline-v0/`](../../../baseline/esp32-baseline-v0/) is
`79f284a` restored verbatim plus zone brackets. Two independent confirmations
that it is the right revision:

- it builds to **RAM 81.7% (267,804 B) / Flash 83.3% (2,621,584 B)**, matching
  [`FLASH_TEST.md`](../FLASH_TEST.md);
- seed 0 gates at **`max_abs = 8.1241e-04`, 0 failures** — the same value as
  [`teammate_esp32_baseline_seed0_v1.log`](../../../baseline/results/teammate_esp32_baseline_seed0_v1.log).

It passes 50/50 host seed-runs against the shared weight blobs, which it
symlinks so both firmwares are measured on byte-identical inputs.

## Status

The tool, both instrumented firmwares, and the host pipeline are complete and
covered by `make check`. **Device captures are still outstanding** — the numbers
above are build-time and host-side facts, which is why nothing here restates a
seconds-per-forward figure. Once boards are free:

```sh
python3 tinyprof/tools/tinyprof.py case2 \
    --port /dev/ttyACM0 --port-v0 /dev/ttyACM1 --seeds 0 1 2 --reps 3
```

writes the artifacts and `report.html` under
`benchmarks/case-02/optimisation/results/tinyprof/`. Entries `00` and `20` should
then gain a "measured by tinyprof" column rather than being rewritten — the two
methods agreeing is the evidence, and overwriting one of them destroys it.

## A note on the host path

The same sources build natively, so the pipeline runs without hardware. On a host
with an FPU the optimised build can measure *slower* than the baseline: the
optimisation trades float work for integer work, which only pays on a core with
no FPU. Host artifacts are stamped `device: host` and the report leads with a
banner refusing to present them as ESP32 results. Structure, call counts, memory
and traffic transfer to the device; times do not.
