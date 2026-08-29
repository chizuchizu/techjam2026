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
| `optimisation/esp32-baseline` (opt23), FAST | 1 C3 | 1.990 s | 1.00x | 5/5 seeds |
| `esp32-cluster-full` | 2 C3s | **1.276 s** | **1.56x** | 4/4 device seeds captured + 25/25 host, 0 failing elements |

Both rows were measured on the *same two boards* in the same session, against
the same opt23 kernels, so the speedup is not confounded by silicon, clock, or
kernel-version differences.

### Single-board reference (this hardware)

`tools/device_test.py /dev/ttyACM0`, FAST mode:

| Seed | s/forward | max_abs | Gate |
|---|---|---|---|
| 0 | 1.991 | 1.03e-3 | PASS |
| 1 | 1.990 | 9.50e-4 | PASS |
| 2 | 1.990 | 1.01e-3 | PASS |
| 3 | 1.990 | 9.80e-4 | PASS |
| 4 | 1.990 | 9.46e-4 | PASS |

Device timing sweep: 1.9880, 1.9885, 1.9884 s. Median **1.990 s**.

### Two-board cluster

Barrier-to-barrier wall time measured by the boards themselves, so it covers
the whole distributed forward on one clock. Host scatter/gather over USB is
outside the window, exactly as the single-board timing excludes its own serial
transfer.

Six back-to-back forwards (`tools/time_cluster.py --reps 6`):

| Rep | Wall (s) | Compute (s) | Link wait (s) |
|---|---|---|---|
| 0 | 1.241 | 1.236 | 0.010 |
| 1 | 1.252 | 1.247 | 0.005 |
| 2 | 1.257 | 1.251 | 0.010 |
| 3 | 1.322 | 1.238 | 0.088 |
| 4 | 1.326 | 1.276 | 0.088 |
| 5 | 1.276 | 1.264 | 0.035 |

**Median 1.276 s**, min 1.241, max 1.326. Time blocked on the link is
**5–88 ms** per forward against roughly 1.6 s of raw transfer, so the exchange
is almost entirely overlapped with arithmetic.

Accuracy, per seed (`tools/run_cluster_e2e.py`). Seed 4 is missing because the
bench host's USB bridge stalled part-way through its output frame, not because
it failed — see the caveat below; the host gate covers all 25 seeds:

| Seed | max_abs | Failing elements | Gate |
|---|---|---|---|
| 0 | 1.04e-3 | 0 | PASS |
| 1 | 9.33e-4 | 0 | PASS |
| 2 | 1.02e-3 | 0 | PASS |
| 3 | 9.96e-4 | 0 | PASS |

The host simulation of the same partition (`tools/shard_host_test`) passes
**25/25 seeds with 0 failing elements**, worst max_abs 1.02e-3, and the
on-device outputs match it.

## Why 1.56x, and why it used to be 1.73x

An earlier build of this cluster, on the opt18-era single-board kernels, ran
3.060 s against 5.293 s — **1.73x**. The partition did not change; the
single-board baseline got much faster underneath it.

opt19–opt23 cut per-token arithmetic about 2.6x (5.293 → 1.990 s) with an
int32 residual stream, LayerNorm fused into the Q15 quantize, a Q15 attention
context at a global scale, and asm GEMM cores. None of that shrinks the work
the row split *cannot* halve:

* **Weight traffic is replicated.** Both boards stream all 24 Q12 weight
  matrices from flash XIP every layer. That cost is per board, not per row, so
  halving the rows does not halve it — and it is now a much larger share of a
  1.99 s forward than of a 5.29 s one.
* **Per-call fixed costs double.** Every GEMM, LayerNorm and quantize is
  invoked twice over half the rows, so amax scans, the per-layer runtime GELU
  LUT rebuild, and the scale arithmetic are all paid twice.
* **Attention is 1.6% imbalanced** by the parity interleave, plus the extra
  per-source logit conversion in the merged softmax.

In absolute terms the cluster is far better than before (1.276 s against the
old 3.060 s); only the ratio to a much stronger baseline is less flattering.
Closing the remaining gap means attacking the replicated weight streaming —
splitting the FFN hidden dimension or the projection output dimension across
boards so each holds half the weights — not the token split.

## Getting the link out of the way

The compute split worked immediately; the transfer did not. Progression on the
same two boards, same partition (measured on the opt18-era kernels, where the
link cost was easiest to see):

(raw capture of that build:
[`two_c3_case2_full_e2e_opt18_v1.csv`](two_c3_case2_full_e2e_opt18_v1.csv))

| Link strategy | Wall time | Time blocked on the link |
|---|---:|---:|
| Blocking TCP, whole 32 KB swapped between pre() and post() | 4.64 s | 1.83 s |
| TCP, streamed per head and overlapped | 5.87–6.62 s | 2.57–3.38 s |
| **UDP with NAK recovery, streamed per head** | **3.06 s** | **0.003–0.032 s** |

Per-forward payload is 131,200 bytes each way (four layers x 32,800 bytes: a
32-byte header plus four per-head chunks of a dequant scale and a Q15 K and V
block).

**TCP was the problem.** A direct probe (`'B'`: eight symmetric 32 KB swaps)
measured **31–79 KB/s** per direction. The Arduino framework ships lwIP with a
5744-byte window and send buffer and no project-level way to raise them, which
caps one connection at roughly `window / RTT`. At that rate the payload costs
more than the compute the split saves, which is why the streamed-TCP version
was *slower than a single board*. Disabling WiFi modem sleep did not help; the
ceiling is the window, not the radio.

UDP has no window. The payload goes out in 1400-byte datagrams; the receiver
NAKs the gaps it still has every 12 ms (suppressed while the stream is still
arriving, so datagrams already in flight are not re-requested) and the sender
retransmits just those. The link is lossy — tens of datagrams per forward are
retransmitted — but recovery is cheap and overlapped.

**Overlap is what makes it free.** The shard releases head `h`'s chunk to a
dedicated FreeRTOS link task the instant it is projected, and blocks on the
peer's head `h` only when attention consumes it, with that head's query
projection computed in between.

## Correctness of the split

Two things the split forces that the single-board path does not have:

* **The global context scale spans both boards.** opt23 emits the attention
  context directly as Q15 at one scale per layer, bounded by the largest |V|
  over every row a token may attend to — which after the split lives on both
  boards. Each node therefore ships its four per-head V bounds in a small
  header ahead of the bulk payload, so the scale is known before the first
  head attends.
* **The two K/V halves keep their own Q15 scales.** Rather than requantizing
  one side to the other's, scores are converted into a shared logit
  fixed-point domain with a per-source multiplier (chosen at runtime so the
  int64 product is safe for the worst-case `|dot| = 32 * 32767^2`), the max is
  taken across both, and the PV products are accumulated per source and
  combined with a single rounding. Cost is one extra multiply per score, which
  the exp path already performed.

## Reproduce

    cd benchmarks/case-02/multiboard/esp32-cluster-full
    make -C tools shard_host_test && ./tools/shard_host_test all   # 25/25, host
    ./tools/flash_boards.sh                                        # both boards
    python3 tools/run_cluster_e2e.py --seeds 0 1 2 3 4             # accuracy
    python3 tools/time_cluster.py --reps 6                         # timing

## Bench caveat

The boards reach this WSL host through `usbipd-win`, whose CDC bridge
intermittently stops delivering bytes — sometimes mid-line — while a board is
busy with the radio. It is a host transport fault: the board has finished the
forward and its reply is still queued, and it arrives once the handle is
recycled. Bus ids are not stable either; flashing or replugging moves a board.

The tooling therefore discovers bus ids, recycles the handle, re-attaches over
usbip, re-resolves which tty belongs to which node, and falls back to the
board's own record of its last forward when the reply is lost. `time_cluster.py`
exists for the same reason: it measures without moving a single 32 KB frame, so
a timing run does not depend on the bridge surviving bulk transfer. A native
Linux or macOS host needs none of this.

## Next

* Four boards: the same parity interleave generalises to `i % N`, and the
  exchange is already addressed by head chunk.
* Cut the replicated weight streaming, which is what now stands between 1.56x
  and 2x.
