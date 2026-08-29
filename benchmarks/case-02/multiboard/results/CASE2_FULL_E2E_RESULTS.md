# Case-2 complete distributed forward on two ESP32-C3 boards

Hardware: two Seeed XIAO ESP32-C3 (160 MHz RV32IMC, no FPU, 400 KB SRAM,
4 MB flash), joined by a direct 2.4 GHz link that node 0 raises itself
(SoftAP + station, no router). Model: the official case-2 shape
`B=1 S=128 D=128 H=4 HD=32 F=128 L=4`, causal, 398,592 fp32 parameters, torch
seed 1234. Gate: `|a-b| <= 0.002 OR |a-b| <= 0.02*|b|` on every one of the
16,384 output elements, against the fp32 torch reference.

Scope: **the complete four-layer body**, both LayerNorms per layer, Q/K/V/O,
causal attention, both residuals, the FFN with GELU, and the final norm. This
is the first multiboard result in this repository that is not a partial-scope
path — see [`../README.md`](../README.md) for the earlier layer-0 and
head-parallel experiments and their exclusions.

## Headline

| Build | Boards | Wall time / forward | Speedup | Gate |
|---|---:|---:|---:|---|
| `optimisation/esp32-baseline`, FAST | 1 C3 | 5.293 s | 1.00x | 5/5 seeds |
| `esp32-cluster-full` | 2 C3s | **3.060 s** | **1.73x** | 5/5 seeds, 0 failing elements |

Both rows were measured on the *same two boards* in the same session, so the
speedup is not confounded by silicon or clock differences.

### Single-board reference (this hardware)

`tools/device_test.py /dev/ttyACM0`, FAST mode:

| Seed | s/forward | max_abs | Gate |
|---|---|---|---|
| 0 | 5.293 | 8.33e-4 | PASS |
| 1 | 5.296 | 9.27e-4 | PASS |
| 2 | 5.293 | 7.22e-4 | PASS |
| 3 | 5.294 | 7.83e-4 | PASS |
| 4 | 5.291 | 6.20e-4 | PASS |

Device timing sweep: 5.2962, 5.2957, 5.2957 s. Median **5.293 s**.

### Two-board cluster

Barrier-to-barrier wall time measured by the boards themselves; host
scatter/gather over USB is outside the window, exactly as the single-board
timing excludes its own serial transfer.

| Seed | Wall (s) | Link wait A / B (s) | K/V moved each way (B) | Retransmitted datagrams | max_abs | Gate |
|---|---|---|---|---|---|---|
| 0 | 3.110 | 0.032 / 0.003 | 131,200 | 48 / 42 | 8.00e-4 | PASS |
| 1 | 3.067 | 0.005 / 0.004 | 131,200 | 32 / 26 | 9.16e-4 | PASS |
| 2 | 3.053 | 0.005 / 0.005 | 131,200 | 24 / 20 | 6.79e-4 | PASS |
| 3 | 3.040 | 0.003 / 0.010 | 131,200 | 24 / 5 | 7.49e-4 | PASS |
| 4 | 3.060 (node B self-measured) | - / 0.004 | 131,200 | - / 21 | 6.20e-4 | PASS |

**Median 3.060 s**, min 3.040, max 3.110. Raw capture:
[`two_c3_case2_full_e2e_v1.csv`](two_c3_case2_full_e2e_v1.csv). Repeated runs
over the session landed in the same 3.04–3.11 s band. Seed 4's wall line was
lost to the USB bridge (see the caveat below); its output was still fetched and
gated, and node B's own measurement of that forward was 3.060 s.

Accuracy is bit-identical to the host simulation of the same partition
(`tools/shard_host_test`), which passes **25/25 seeds with 0 failing
elements**, worst max_abs 9.2e-4 — marginally *better* than the single-board
firmware's 9.4e-4, because each K/V half keeps its own quantization scale.

## Why 1.73x and not 2x

Per-node arithmetic is 3.03–3.11 s against 5.293 s on one board, so the
compute itself carries essentially all of the speedup. The gap to 2x is not communication (the
measured link wait is 1–24 ms per forward) — it is work the split does not
halve:

* **Weight traffic is replicated.** Both boards stream all 24 Q12 weight
  matrices from flash XIP every layer. That traffic is per-board, not per-row,
  and the GEMM kernels sit right at the memory/compute ridge, so halving the
  rows does not halve the time spent fetching weights.
* **Per-call fixed costs double.** Every GEMM, LayerNorm and quantize is now
  invoked twice over half the rows, so amax scans, LUT builds (the runtime
  GELU LUT is rebuilt per layer *per board*) and scale arithmetic are paid
  twice.
* **Attention is 1.6% imbalanced** by the parity interleave, plus the extra
  per-source logit conversion in the merged softmax.

## Getting the link out of the way

The compute split worked immediately; the transfer did not. Progression on the
same two boards, same partition:

| Link strategy | Wall time | Time blocked on the link |
|---|---:|---:|
| Blocking TCP, whole 32 KB swapped between pre() and post() | 4.64 s | 1.83 s |
| TCP, streamed per head and overlapped | 5.87–6.62 s | 2.57–3.38 s |
| **UDP with NAK recovery, streamed per head** | **3.06 s** | **0.003–0.032 s** |

Per-forward payload is 131,200 bytes each way (four layers x 32,800 bytes:
four per-head chunks of two dequant scales plus a Q15 K and V block).

**TCP was the problem.** A direct probe (`'B'`: eight symmetric 32 KB swaps)
measured **31–79 KB/s** per direction. The Arduino framework ships lwIP with a
5744-byte window and send buffer and no project-level way to raise them, which
caps one connection at roughly `window / RTT`. At that rate the 128 KB per
forward costs more than the 2.2 s of compute the split saves, which is why the
streamed-TCP version was *slower than a single board*. Disabling WiFi modem
sleep did not help; the ceiling is the window, not the radio.

UDP has no window. The payload goes out in 1400-byte datagrams; the receiver
NAKs the gaps it still has every 12 ms (suppressed while the stream is still
arriving, so datagrams already in flight are not re-requested) and the sender
retransmits just those. Observed retransmission is 5–48 datagrams out of 96
per forward — the link is lossy, but recovery is cheap and, crucially,
overlapped.

**Overlap is what makes it free.** The shard releases head `h`'s chunk to a
dedicated FreeRTOS link task the instant it is projected, and blocks on the
peer's head `h` only when attention consumes it — with that head's query
projection computed in between. Raw transfer is roughly 1.6 s per forward;
measured *waiting* is 1–24 ms.

## Correctness of the split

The two K/V halves arrive with different Q15 dequant scales. Requantizing one
side to the other's would lose a bit; instead attention converts each side's
integer dot product into a shared logit fixed-point domain with a per-source
multiplier chosen at runtime (`M` in `[2^26, 2^27)` with a matching shift, so
`dot * M` stays inside int64 for the worst-case `|dot| = 32 * 32767^2`), takes
the max across both, and accumulates the PV product separately per source
before a single fp32 combine. Cost is one extra multiply per score, which the
existing exp path already performed — so the merged softmax is no more
expensive than the single-board one, and slightly more accurate.

## Reproduce

    cd benchmarks/case-02/multiboard/esp32-cluster-full
    make -C tools shard_host_test && ./tools/shard_host_test all   # 25/25, host
    ./tools/flash_boards.sh                                        # both boards
    python3 tools/run_cluster_e2e.py --seeds 0 1 2 3 4

## Bench caveat

The boards reach this WSL host through `usbipd-win`, whose CDC bridge
intermittently stops delivering bytes — sometimes mid-line — while a board is
busy with the radio. It is a host transport fault: the board has finished the
forward and its reply is still queued, and it arrives once the handle is
recycled. The driver recycles, re-attaches over usbip, re-resolves which tty
belongs to which node, and falls back to "timing unavailable" for a seed whose
timing line was lost while still validating that seed's output. Some rows above
therefore carry one node's self-measured time rather than both. A native Linux
or macOS host needs none of this.

## Next

* Four boards: the same parity interleave generalises to `i % N`, and the
  exchange is already addressed by head chunk. The K/V a node must receive
  grows with `N-1`, so the payload per node stays 32 KB per layer while the
  compute keeps shrinking.
* Cut the replicated weight traffic, which is what stands between 1.73x and 2x
  — a shared blocked-GEMM schedule, or splitting the FFN hidden dimension
  across boards so each holds half of `f1`/`f2`.
