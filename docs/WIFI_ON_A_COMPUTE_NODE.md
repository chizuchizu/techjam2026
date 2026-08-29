# Brief: getting WiFi onto a compute node

**Status: blocked, measured.** A board can host either a full-sequence forward
or the TCP/IP stack, not both. This document states the problem, the numbers,
what has already been tried, and the directions that remain, so the work can be
picked up without repeating any of it.

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

Target: 8 boards, `ceil(B/N) * 1.99 s`, i.e. B=128 in ~31.8 s.

---

## 2. The blocker, with numbers

Board: Seeed XIAO ESP32-C3. 400 KB SRAM, of which **`dram0_0_seg` = 327,680 B**
is available to us. 4 MB flash. No PSRAM.

| Build | Static DRAM | Free for heap |
|---|---:|---:|
| Full-sequence node, no WiFi (`esp32-baseline`) | 274,564 B | 53,116 B |
| Half-sequence node **with** WiFi (`esp32-cluster`) | 221,916 B | 105,764 B |

Adding WiFi to the full-sequence node fails to link:

```
region `dram0_0_seg' overflowed by 32384 bytes
```

From that overflow, **the WiFi + lwIP stack costs ~85,500 B of static DRAM**.
At runtime it wants more: the cluster firmware reports 105,764 B free before
`WiFi.begin()` and **~36,800 B after**, so **~69 KB of heap** on top.

So a full-sequence node needs roughly `274,564 + 85,500 + 69,000 = 429 KB`
against 328 KB available — **short by about 100 KB**, out of a 274 KB arena.
This is not a tuning problem; it is a quarter of the model's working memory.

**Why the cluster firmware gets away with it:** it holds only half the sequence
(64 of 128 token rows), so its arena is 127 KB instead of 274 KB. WiFi on a
*sharded* node is proven and works today. WiFi on a *full-forward* node is the
open problem — and data parallelism needs the full forward on every board.

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

## 5. Directions that remain

Ranked by expected value. None is proven; the numbers are estimates and should
be measured, not trusted.

### A. Sequence tiling of the forward — *most promising*
Process the forward in row tiles so activation memory scales with the tile, not
with `S=128`. Attention still needs all K and V resident (~64 KB), but the rest
could drop to a tile's worth. Plausible arena: **~120 KB**, which leaves room
for WiFi untrimmed.

The real argument for it is not WiFi: **cases 13 and 14 (S=1024 and
S=100,000) are impossible without it**, and they are on the roadmap. Adjacent
work already exists — see case-09's "5-slot union" and case-10's "overlay
per-head Q/K buffers onto dead scratch" commits.

Cost: a week-ish, touches the validated kernel path, needs the full gate at
every step.

### B. Custom `sdkconfig` — necessary but not sufficient
Requires migrating to `framework = espidf` with Arduino as a component; the
PlatformIO Arduino framework ships a **precompiled** IDF whose config is not
reachable. Then: `ESP_WIFI_STATIC_RX_BUFFER_NUM` 10 -> 3 (~11 KB),
`ESP_WIFI_DYNAMIC_RX/TX_BUFFER_NUM` 32 -> 8 (~10-15 KB), `ESP_WIFI_AMPDU_RX`
off, `LWIP_IPV6` off (~10 KB), ESP-MESH off (it is compiled in — `g_mesh_*`
symbols are in the map), `LWIP_MAX_SOCKETS` 10 -> 4.

Estimated **40-60 KB off 85 KB**. Combined with a FAST-only build (shrinking
`g_buf1` from 64 KB to the 32 KB `v_all` actually needs) it lands at roughly
**327 KB against 328 KB** — passing by nothing, on optimistic assumptions,
having given up EXACT mode. Do not pursue alone; only as a multiplier on A.

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

## 6. How to check a proposal quickly

```bash
# memory budget of any env (watch the RAM line, and the overflow message)
cd benchmarks/case-02/optimisation/esp32-baseline && pio run -e <env>

# host accuracy gate — do this before flashing
make -C tools host_test && ./tools/host_test all          # 25/25 expected
cd ../../multiboard/esp32-cluster-full
make -C tools shard_host_test && ./tools/shard_host_test all
```

Largest static symbols come from `.pio/build/<env>/firmware.map`.

## 7. Key files

| Path | What |
|---|---|
| `benchmarks/case-02/optimisation/esp32-baseline/src/model.c` | the arena; opt23 FAST path |
| `.../src/kernels.c` | `a16` scratch, asm GEMM cores |
| `benchmarks/case-02/multiboard/esp32-cluster-full/src/link.cpp` | working WiFi on a *sharded* node: SoftAP, UDP + NAK, async exchange |
| `benchmarks/case-02/multiboard/esp32-cluster-full/src/model_shard.c` | half-sequence arena, 127 KB |
| `benchmarks/batch-dp/tools/run_batch_dp.py` | N-board data-parallel runner |
