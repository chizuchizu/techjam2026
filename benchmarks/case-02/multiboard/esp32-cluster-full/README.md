# Case-02 two-board cluster — complete distributed forward

The full case-2 Transformer body (`B=1 S=128 D=128 H=4 F=128 L=4`, causal) run
end to end across **two Seeed XIAO ESP32-C3 boards**, validated against the
torch reference with the benchmark's own gate.

This is the first complete distributed case-2 result in the repository: every
operator of all four layers runs on the pair, not a single layer or a subset of
the heads. See [`../README.md`](../README.md) for the earlier partial-scope
multiboard experiments.

| | Boards | Wall time | Speedup | Gate |
|---|---:|---:|---:|---|
| Optimised single-board firmware | 1 C3 | 5.293 s | 1.00x | 5/5 seeds |
| This cluster | 2 C3s | **3.060 s** | **1.73x** | 5/5 seeds, 0 failing elements |

Both rows are the same two physical boards at 160 MHz, same weights, same
kernels. Full numbers and the derivation of the partition are in
[`../results/CASE2_FULL_E2E_RESULTS.md`](../results/CASE2_FULL_E2E_RESULTS.md).

## How the model is split

Global token row `i` lives on node `i % 2`, at local index `i / 2`.

Every operator in the case-2 body acts on one token at a time — LayerNorm, the
Q/K/V/O projections, both residuals, the FFN, GELU, the final norm — so each
board simply runs them over its own 64 rows. The single exception is causal
attention, where row `i` needs the keys and values of every `j <= i`. That
makes the *only* inter-board traffic one K/V exchange per layer.

Interleaving by parity rather than splitting the sequence in half balances the
causal triangle: node 0 accumulates `sum over even i of (i+1)` = 4096 score
rows against node 1's 4160, a 1.6% imbalance. A contiguous split would have
been 2080 against 6176.

The two K/V halves keep their own Q15 dequant scales. Rather than requantizing
one side to the other's scale, attention converts each side's integer dot
product into a common logit fixed-point domain with a per-source multiplier, so
the shard is never less accurate than the single-board path — the measured
per-seed max error is in fact slightly lower.

## How the transfer hides

Per layer each board ships 32,800 bytes (four per-head chunks of two dequant
scales plus a Q15 K and V block) and receives the same, 131,200 bytes each way
per forward.

The payload is streamed rather than exchanged in one block: the shard hands
head `h` to the link the instant it is projected, and only blocks on the peer's
head `h` at the moment attention consumes it, with the query projection for
that head computed in between. A dedicated FreeRTOS task keeps the socket busy
throughout. Measured time actually spent *waiting* on the link is **3–32 ms per
forward**, against ~1.6 s of raw transfer — it is essentially all overlapped.

The bulk payload goes over **UDP** with NAK-based recovery, not TCP. The
Arduino framework ships lwIP with a 5744-byte window that cannot be raised
from a project build, which caps one TCP connection at a measured 31–79 KB/s
and made the first working version *slower* than one board. See the results
report for that progression.

The two boards form their own network — node 0 raises a SoftAP, node 1 joins
it — so no router, credentials, or infrastructure are involved.

## Layout

    platformio.ini          one env; the same binary runs both nodes
    partitions_cluster.csv  3.5 MB app (the ~2.4 MB weight blobs + WiFi)
    src/model_shard.h/.c    the sharded forward (pre / exchange / post)
    src/link.h/.cpp         SoftAP peer link, async UDP exchange, barriers
    src/main.cpp            serial protocol, role assignment, timing
    tools/shard_host_test.c host validation of the distributed numerics
    tools/run_cluster_e2e.py host coordinator and gate
    tools/attach_boards.sh  re-attach the boards to WSL over usbip
    tools/flash_boards.sh   build, flash both, re-attach
    weights*.bin            symlinks to ../../optimisation/esp32-baseline

The numeric kernels are compiled straight out of
`../../optimisation/esp32-baseline/src/kernels.c` and the weight blobs are that
project's, so the cluster cannot drift from the optimised single-board path.

## Validate on the host first

The two nodes and their exchange run in one process, so the distributed
numerics are checked before anything is flashed:

    make -C tools shard_host_test && ./tools/shard_host_test all

25/25 seeds pass, 0 failing elements, worst max_abs 9.2e-4.

## Run on hardware

    ./tools/flash_boards.sh                       # build, flash both, re-attach
    python3 tools/run_cluster_e2e.py --seeds 0 1 2 3 4 --reps 3

The driver assigns the roles, scatters the input rows, triggers one
barrier-bracketed distributed forward, gathers the output rows and applies the
gate (`|a-b| <= 0.002 OR |a-b| <= 0.02*|b|`). Ports default to `auto`.

The reported wall time is the boards' own barrier-to-barrier measurement, so it
covers the whole distributed forward on one clock. Host scatter/gather over USB
is outside it, exactly as the single-board `R` timing excludes its own serial
transfer.

### A note on the USB bridge

On this bench the boards reach WSL through `usbipd-win`, and that bridge
intermittently stops delivering CDC bytes — sometimes mid-line — while a board
is busy with the radio. It is a host transport problem, not a device one: the
board has already finished the forward and its reply is still queued.

The driver therefore recycles the handle, re-attaches over usbip and
re-resolves which tty belongs to which node (the numbering changes), and treats
a lost timing line as "timing unavailable" while still validating the output,
which `O` re-streams on demand. A native-Linux or macOS host does not need any
of this.

The worker has no authentication and is meant for a trusted benchmark bench.
