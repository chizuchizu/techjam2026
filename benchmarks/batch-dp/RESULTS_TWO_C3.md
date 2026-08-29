# Batch cases on two ESP32-C3 boards — data parallel

Hardware: two Seeed XIAO ESP32-C3 at 160 MHz, each running the maintained
single-board firmware from
[`../case-02/optimisation/esp32-baseline`](../case-02/optimisation/esp32-baseline)
(opt23) with the shared case-2 weight blobs. No inter-board link exists in this
decomposition — input `i` simply runs on board `i % N`.

Gate: `|a-b| <= 0.002 OR |a-b| <= 0.02*|b|` on all 16,384 elements of every one
of the B outputs, against the fp32 torch reference for that case's batch.

## Results

| Case | B | Boards | Per-forward | One board | Two boards | Speedup | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| [3](../case-03/) | 4 | 2 | 1.990 s | 8.0 s | **4.0 s** | **2.00x** | PASS, 4/4 forwards, 0 failing elements |
| [4](../case-04/) | 16 | 2 | 1.990 s | 31.8 s | **15.9 s** | **2.00x** | PASS, 16/16 forwards, 0 failing elements |
| [1](../case-01/) | 64 | 2 | 1.990 s | 127.4 s | **63.7 s** | **2.00x** | PASS, 64/64 forwards, 0 failing elements |
| [5](../case-05/) | 128 | 2 | 1.990 s | 254.8 s | **127.4 s** | **2.00x** | PASS, 128/128 forwards, 0 failing elements |

All 212 forwards across the four cases were gated individually against the
torch reference for that batch; worst max_abs 1.34e-3 against a 2e-3 tolerance.

Per-forward is the board's own measurement of one input; *one board* is what a
single board would take for all B inputs; *two boards* is the max over boards
of its summed forward times. Host serial transfer is excluded from both sides,
the same convention the single-board and case-2 cluster numbers use.

**The speedup is exactly 2.00x** because the decomposition is embarrassingly
parallel: the boards share no state, exchange nothing, and B divides evenly by
2 in every case here. The only way to lose efficiency is an odd remainder —
with N boards the batch time is `ceil(B/N) * t_forward`, so B=4 on 8 boards
would idle half the fleet rather than go faster.

## End-to-end, including host transfer

| Case | B | Compute | End-to-end | Transport share |
|---:|---:|---:|---:|---:|
| 3 | 4 | 4.0 s | 8.1 s | 51% |
| 4 | 16 | 15.9 s | 32.4 s | 51% |
| 1 | 64 | 63.7 s | 129.4 s | 51% |
| 5 | 128 | 127.4 s | 259.0 s | 51% |

Each input costs a 64 KB frame down and a 64 KB frame up over USB CDC. The
host paces the downstream at 1 KB / 20 ms to stay inside the C3's CDC receive
limit, which alone is ~1.3 s per input against 1.99 s of compute. That is a
property of the bench harness, not of the decomposition — a real deployment
would not feed the boards one 64 KB frame at a time over a serial console —
but it does mean the *measured wall clock* of a batch run is roughly twice its
compute time.

## Note on the USB bridge

Unlike the case-2 cluster runs, these did not suffer CDC stalls: cases 1/3/4/5
never turn the radio on, and the stalls on this bench track WiFi activity. The
one failure mode seen here is a board left blocked inside `read_input()` after
an interrupted run, which the coordinator now clears by completing the pending
frame with zeros.

The coordinator refuses to report a speedup for a batch where any input was
lost, so a partial run is visible as INCOMPLETE rather than as a faster batch.

## Reproduce

```bash
python3 tools/export_batch.py --batch 4 16 64 128
python3 tools/run_batch_dp.py --batch 4 16 64 128
```
