# Challenge and solution: fitting WiFi beside a full C3 forward

**Status: implemented and physically verified on two ESP32-C3 boards.** The
opt-in `esp32-wifi-tiled` build hosts both a complete FAST forward and the
Arduino WiFi/TCP stack. It row-tiles the projection/FFN work and processes
attention heads sequentially, reducing the working arena enough to associate
with WiFi and execute repeated forwards. The default published firmware is
unchanged.

---

## 1. Why we want this

The data-parallel batch benchmark (cases 1, 3, 4, 5 — see
[`../benchmarks/batch-dp/`](../benchmarks/batch-dp/)) scales at exactly **2.00x
on two boards** and would scale linearly to eight. The blocker is not compute
and not the algorithm: it is **host connectivity**. The bench host has two
usable USB ports, so only two boards can be driven at once.

Over WiFi the host would reach every board simultaneously and USB would be
needed only for flashing. Secondary win: the paced USB CDC frame (1 KB / 20 ms,
~1.3 s per 64 KB input) is **about half the end-to-end wall time** of a batch
run, and it is also the single flakiest component in the whole harness.

The original latency target assumed the 1.99 s untiled forward. The
memory-saving forward is slower at 4.214 s median on device, so WiFi solves
host connectivity but does not preserve that latency target.

---

## 2. The original blocker and the implemented result

Board: Seeed XIAO ESP32-C3. 400 KB SRAM, of which **`dram0_0_seg` = 327,680 B**
is available to us. 4 MB flash. No PSRAM.

| Build | Static DRAM | Free for heap |
|---|---:|---:|
| Full-sequence node, no WiFi (`esp32-baseline`) | 274,564 B | 53,116 B |
| Half-sequence node **with** WiFi (`esp32-cluster`) | 221,916 B | 105,764 B |
| Tiled full-forward node **with real WiFi path linked** (`esp32-wifi-tiled`) | **173,060 B** | **154,620 B nominal** |

Adding WiFi to the original full-sequence node failed to link:

```
region `dram0_0_seg' overflowed by 32384 bytes
```

From that overflow, **the WiFi + lwIP stack costs ~85,500 B of static DRAM**.
At runtime it wants more: the cluster firmware reports 105,764 B free before
`WiFi.begin()` and **~36,800 B after**, so **~69 KB of heap** on top.

So the original full-sequence node needed roughly
`274,564 + 85,500 + 69,000 = 429 KB` against 328 KB available — **short by
about 100 KB**, out of a 274 KB arena.

The tiled build cuts static DRAM by **101,504 B even after linking the active
WiFi code**. PlatformIO reports 154,620 B free for heap. Physical boot capture
on both boards measured 145,004 B before enabling WiFi, 100,048 B after station
mode, 99,808 B after `WiFi.begin()`, and **98,380 B after association and TCP
server startup**. The associated radio therefore consumes 46,624 B of the
available heap on this image, leaving a large measured margin.

**Why the cluster firmware gets away with it:** it holds only half the sequence
(64 of 128 token rows), so its arena is 127 KB instead of 274 KB. WiFi on a
*sharded* node was already proven. WiFi on a *full-forward* node was the open
problem — and data parallelism needs the full forward on every board.

### Where the 274 KB goes

From `benchmarks/case-02/optimisation/esp32-baseline/src/model.c` (opt23):

| Buffer | Size | Live when |
|---|---:|---|
| `g_x` (int32 residual stream) | 64 KB | whole forward |
| `g_buf1` | 64 KB | `v_all` (Q15, all heads) uses 32 KB of it in FAST |
| `g_buf2` | 64 KB | `g_ctxq` (Q15 ctx, 32 KB) **and** the FFN1 int32 scratch (64 KB) |
| `a16` (kernels.c, Q15 activations) | 32 KB | whole forward |
| `g_qh`, `g_kh`, `g_acc`, misc | ~40 KB | |

The obvious overlaps are **already taken**: `g_ctxq` and the FFN1 scratch both
alias `g_buf2`; `v_all` aliases `g_buf1`. Do not propose these as savings.

---

## 3. Already tried — do not repeat

| Attempt | Result |
|---|---|
| Add `WiFi.h` + TCP server to the full-forward firmware | **Fails to link**, 32,384 B over. Code was written and removed; the `Stream*` refactor that lets one command handler serve both USB and TCP is the reusable part (`HWCDC` and `WiFiClient` both derive from `Stream`, and `Print::printf` exists on both). |
| TCP between boards for the cluster K/V exchange | **31–79 KB/s.** Arduino ships lwIP with a 5744-byte window and send buffer, unreachable from a project build, capping one connection at `window / RTT`. This is why the cluster's bulk transfer runs over **UDP with NAK recovery** instead. |
| Disabling WiFi modem sleep (`WiFi.setSleep(false)`) | No help. The ceiling is the window, not the radio. |
| ESP-NOW (measured on this bench, 2 boards) | **~60 KB/s.** A 64 KB input would take ~1 s — no better than the serial we are trying to escape. |
| Teammate PC-master ESP-NOW relay | **Not a compute path.** The checked-in `W N` command generates and echoes one 1–240 B pattern between two link-only images. It has no arbitrary host payload, tensor fragmentation, retries, model invocation, or N-node routing. |
| Dedicated WiFi–UART radio sidecar | **Implemented, build-verified, not yet physically measured.** One link-only C3 streams TCP port 5000 to one radio-free opt23 C3 over 2 Mbaud UART. It preserves the 1.99 s compute path but costs two boards per logical worker. See [`../esp32-linkbench/docs/PC_MASTER_WIFI_BRIDGE.md`](../esp32-linkbench/docs/PC_MASTER_WIFI_BRIDGE.md). |

---

## 4. Invariants any proposal must respect

1. **The accuracy gate is not negotiable.** `|a-b| <= 0.002 OR |a-b| <= 0.02*|b|`
   on all 16,384 output elements. Run `make -C tools host_test && ./tools/host_test all`
   (25/25, from `benchmarks/case-02/optimisation/esp32-baseline/tools/`) and the
   shard equivalent in `benchmarks/case-02/multiboard/esp32-cluster-full/tools/`.
   **Validate on the host before flashing anything** — that is how the two-board
   shard landed correct on its first device run.
2. **Do not silently change the measured firmware.** Anything optional goes
   behind a build flag with its own env, as `TM_SERVO_PIN` does. Published
   numbers must stay reproducible from the default envs.
3. **Timing convention:** host transfer is excluded from every reported number,
   single-board and multiboard alike. Keep it that way, and report end-to-end
   separately.
4. **Report partial runs as partial.** The batch runner refuses to compute a
   speedup when any input was lost; do not weaken that.

---

## 5. Implemented design and follow-ons

### A. Row-tiled, head-sequential FAST forward — implemented

`src/model_tiled.c` uses 16-row tiles and retains only the current attention
head's K/V. The major model workspaces are:

| Workspace | Bytes |
|---|---:|
| input / residual / final-output union | 65,536 |
| full Q15 attention context | 32,768 |
| current-head K + V | 16,384 |
| tile accumulator + head output | 9,216 |
| attention scores/probabilities | 1,536 |
| shared kernel Q15 tile | 4,096 |

The forward first sweeps V tiles to choose one safe context scale. It then
builds K/V for one head, streams Q by tile through causal attention, and moves
to the next head. Only after all four heads have filled the context does the O
projection mutate the residual. Norm2, FFN1, GELU, and FFN2 then run one tile
at a time.

This is deliberately FAST-only and opt-in. The full host gate passes **25/25**
seeds with worst maximum absolute error **1.0778e-3**. The default FAST/EXACT
build still passes 50/50 seed-runs. The physical two-board TCP gate also passes
**25/25** seeds with zero failing elements and worst maximum absolute error
**1.2370e-3**.

Scope limit: this solves the `S=128` WiFi fit, but it is **not O(tile) in
sequence length**. The residual, context, and current-head K/V remain O(S).
Cases 13 and 14 still require external storage/recomputation plus a genuinely
streaming or FlashAttention-style schedule.

### B. Custom `sdkconfig` — necessary but not sufficient
Requires migrating to `framework = espidf` with Arduino as a component; the
PlatformIO Arduino framework ships a **precompiled** IDF whose config is not
reachable. Then: `ESP_WIFI_STATIC_RX_BUFFER_NUM` 10 -> 3 (~11 KB),
`ESP_WIFI_DYNAMIC_RX/TX_BUFFER_NUM` 32 -> 8 (~10-15 KB), `ESP_WIFI_AMPDU_RX`
off, `LWIP_IPV6` off (~10 KB), ESP-MESH off (it is compiled in — `g_mesh_*`
symbols are in the map), `LWIP_MAX_SOCKETS` 10 -> 4.

Estimated **40-60 KB off 85 KB**. It is no longer required for case 2 because
the tiled Arduino build has adequate margin. It may still be useful together
with a future long-sequence streaming design.

### C. Different hardware
ESP32-S3: 512 KB SRAM plus PSRAM. Fits both comfortably. Worth pricing before
spending a week on A.

### D. Sidestep WiFi entirely — cheapest working answers
- **Powered USB hub.** ~$15, zero code. `run_batch_dp.py` already takes
  `--boards` and globs every `/dev/ttyACM*`; `flash_boards.sh` and
  `attach_boards.sh` honour `TM_BOARDS=8`. True 8.0x on cases 1/4/5 today.
- **UART daisy chain.** A UART link costs **~2 KB of RAM** against WiFi's
  ~150 KB, and at 5 Mbaud carries ~500 KB/s — *ten times faster* than the paced
  USB CDC. One board sustains 128 KB / 1.99 s = 64 KB/s of host traffic, so a
  500 KB/s link supports ~8 boards. Use **two chains of four** rooted at the two
  USB ports (halves head-link load); **not a ring** — in a ring every byte
  traverses all N hops, so link load is N x worse.
- **One-wire barrier.** For a *timing* run, nothing needs to move: per-forward
  compute is input-independent (measured 1.990-1.992 s across every seed). Arm
  each board over USB two at a time, then use two shared GPIOs — a GO line and
  an open-drain wired-AND DONE line — to start all boards together and measure
  `ceil(B/N) * t_forward` on one clock. Three wires, ~0 RAM, scales to 64
  boards. Measures batch **compute** time, which is the convention already in
  use; per-input data movement is not exercised, and should be stated.

---

## 6. Physical multi-board results

Both boards joined the same 2.4 GHz WPA2 LAN and served persistent TCP on port
5000. A case-3 `B=4` run assigned two independent forwards to each board:

| Measurement | Result |
|---|---:|
| Inputs completed | 4/4, none missing |
| Accuracy failures | 0 elements |
| Worst maximum absolute error | 1.050e-3 |
| Median forward | 4.218 s |
| One-board compute equivalent | 16.870 s |
| Two-board compute wall | 8.437 s |
| Compute speedup | **2.00x** |
| End-to-end including WiFi TCP | 9.9 s |

The complete 25-seed case-2 device gate was then distributed across the same
two nodes: **25/25 passed**, worst max-absolute error `1.2370e-3`, median
forward `4.214 s`, wall time `64.5 s`. The reproducible B=4 result is stored in
[`../benchmarks/batch-dp/results_two_c3_wifi_tiled.json`](../benchmarks/batch-dp/results_two_c3_wifi_tiled.json).

Two additional C3s were then flashed with the identical image. With B=4 and
one forward assigned to each of four boards, all 4/4 outputs passed with no
losses. Compute wall fell to **4.215 s**, exactly **4.00x** versus the measured
16.859 s single-board equivalent; WiFi-inclusive wall time was 6.5 s. See
[`../benchmarks/batch-dp/results_four_c3_wifi_tiled.json`](../benchmarks/batch-dp/results_four_c3_wifi_tiled.json).

The important tradeoff is visible: tiling removes the memory blocker and WiFi
transport gives exact 2-board scaling, but repeated normalization/projection
sweeps increase a forward from ~1.99 s to ~4.21 s. Future optimization should
recover that compute cost without restoring the full arena.

### Fair comparison with the original optimized forward

The original opt23 forward measures 1.9904 s; the tiled forward measures
4.2147 s. Tiling is therefore **2.12x slower per forward** (+111.7% latency,
47.2% of the original throughput). The reported 4.00x result is scaling versus
one *tiled* node, not versus the faster opt23 firmware.

For the same B=4 workload:

| Configuration | Compute wall | End-to-end |
|---|---:|---:|
| Original opt23, 1 board | 7.962 s | not measured in this run |
| Original opt23, 2 boards over USB | **3.981 s** | 8.1 s |
| Tiled WiFi, 2 boards | 8.437 s | 9.9 s |
| Tiled WiFi, 4 boards | **4.215 s** | **6.5 s** |

Thus four tiled boards are 1.89x faster in compute than one original board for
B=4. Against two original USB boards, four tiled boards are 5.9% slower in
compute but 19.8% faster end-to-end because WiFi removes the paced USB cost.
A hypothetical four-board opt23 setup using a powered USB hub would still win
on compute at about 1.99 s; the tiled design's value is wireless scalability
and SRAM feasibility, not single-board speed.

In competition terms: the WiFi-capable design pays a substantial per-node
overhead, primarily from the tiled schedule needed to make WiFi fit. Once the
batch contains enough independent inputs, compute throughput scales linearly
with node count—verified at 2.00x on two nodes and 4.00x on four. Steady-state
aggregate throughput exceeds one original opt23 board from three tiled nodes
onward (`N / 4.214 > 1 / 1.990`); for the discrete B=4 case, four nodes are
needed because three nodes leave one node executing two forwards. End-to-end
scaling may eventually be limited by shared AP bandwidth, so linearity beyond
the measured four nodes should be tested rather than assumed.

## 7. How to check a proposal quickly

```bash
# memory budget of any env (watch the RAM line, and the overflow message)
cd benchmarks/case-02/optimisation/esp32-baseline && pio run -e <env>

# host accuracy gate — do this before flashing
make -C tools host_test_tiled && ./tools/host_test_tiled all --fast
cd ../../multiboard/esp32-cluster-full
make -C tools shard_host_test && ./tools/shard_host_test all

# build the real WiFi path (without secrets.h it intentionally builds USB-only)
cd ../../optimisation/esp32-baseline
cp secrets.example.h secrets.h       # fill in benchmark-LAN credentials
pio run -e esp32-wifi-tiled
```

Largest static symbols come from `.pio/build/<env>/firmware.map`.

## 8. Key files

| Path | What |
|---|---|
| `benchmarks/case-02/optimisation/esp32-baseline/src/model.c` | the arena; opt23 FAST path |
| `.../src/model_tiled.c` | opt-in row-tiled/head-sequential FAST forward |
| `.../src/main_wifi.cpp` | persistent TCP server and shared USB/TCP command protocol |
| `.../src/kernels.c` | `a16` scratch, asm GEMM cores |
| `benchmarks/case-02/multiboard/esp32-cluster-full/src/link.cpp` | working WiFi on a *sharded* node: SoftAP, UDP + NAK, async exchange |
| `benchmarks/case-02/multiboard/esp32-cluster-full/src/model_shard.c` | half-sequence arena, 127 KB |
| `benchmarks/batch-dp/tools/run_batch_dp.py` | N-board data-parallel runner |

Case 2 is now host-, linker-, heap-, accuracy-, transport-, and four-board
scaling-verified. Remaining work is the 8-board run and the separate
long-sequence design for cases 13/14.
