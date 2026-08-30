# opt 24 — row-tiled FAST forward for a full WiFi node

## Goal

The opt23 full forward uses 274,564 B static DRAM. Linking Arduino WiFi/lwIP
adds enough static state to overflow the C3 before its then-estimated ~69 KB
runtime heap is considered. Data-parallel batch nodes cannot use the existing half-sequence
cluster arena because every node must execute a complete forward.

This optimization is an opt-in memory schedule. It does not change the default
firmware or the checked-in timing result.

## Schedule

`TM_TILE_ROWS=16` reduces LayerNorm, projection-accumulator, Q, and FFN
workspaces to one row tile. The persistent state is:

| buffer | bytes | lifetime |
|---|---:|---|
| `g_x` int32/fp32 union | 65,536 | input, residual, final output |
| `g_ctxq` | 32,768 | complete Q15 context for O projection |
| `g_kh`, `g_vh` | 8,192 each | current attention head |
| tile accumulator | 8,192 | current projection/FFN tile |
| tile head output | 1,024 | current Q or V projection |
| scores + probabilities | 1,536 | one causal attention row |
| kernel `a16` | 4,096 | current normalized/FFN activation tile |

For each transformer layer:

1. Sweep normalized V projections by tile/head to find one conservative
   context bound.
2. For one head, build all K/V tiles and requantize them to head-global scales.
3. Project one Q tile at a time and immediately execute its causal attention
   rows into that head's slice of `g_ctxq`.
4. Repeat for all heads. Only then run the complete O projection and update the
   residual.
5. Run norm2, FFN1, GELU, and FFN2 per tile.

One layer-global context scale preserves the existing full O-projection
kernel. K/V need one additional requantization rounding; Q can retain a
tile-local scale because all logits for a query row share it.

## Correctness and safety

- Tiled FAST host gate: 25/25 seeds pass, worst maximum absolute error
  `1.0778e-3` against the competition OR gate.
- Default FAST + EXACT host gate: 50/50 seed-runs still pass.
- AddressSanitizer and UBSan smoke test: clean.
- A dedicated 1 KB `int16_t` tile output avoids aliasing the `int32_t`
  accumulator; the first prototype's logically ordered type-punning was still
  strict-aliasing undefined behavior.
- Compile-time checks require a tile divisible by 8 and `TM_F <= TM_D`, which
  are the constraints of the shared reduced `a16` kernel workspace.

## Memory result

With real credentials present so the WiFi-start path cannot be dead-stripped:

| image | static DRAM | flash |
|---|---:|---:|
| default `esp32-baseline` | 274,564 B | 2,644,430 B |
| `esp32-wifi-tiled` | **173,060 B** | **3,120,460 B** |

The actual DRAM linker segment is 321,296 B, leaving 148,236 B after static
allocation. Both physical boards measured 145,004 B free before WiFi and
**98,380 B free after association and TCP server startup**. The dedicated
3.5 MB app partition leaves 549,572 B of flash.

## Physical result

- Distributed case-2 device gate: 25/25 seeds pass, zero failing elements,
  worst `max_abs=1.2370e-3`, median `4.214 s/forward`.
- Two-board case-3 B=4 over TCP: 4/4 forwards pass, no missing inputs,
  `2.00x` compute speedup, `8.437 s` compute wall, `9.9 s` end-to-end.
- Four-board case-3 B=4 over TCP: 4/4 forwards pass, no missing inputs,
  `4.00x` compute speedup, `4.215 s` compute wall, `6.5 s` end-to-end.
- Result artifacts: `benchmarks/batch-dp/results_two_c3_wifi_tiled.json` and
  `benchmarks/batch-dp/results_four_c3_wifi_tiled.json`.

The extra V-bound, normalization, and per-head tile sweeps make the tiled
forward slower than opt23 (~4.21 s versus ~1.99 s). This is the primary cost
of resolving the memory conflict.

## Scope and remaining work

This is row-tiled and head-sequential, not O(tile) in S: `g_x`, `g_ctxq`, and
the current-head K/V still scale with sequence length. It solves the case-2
WiFi fit but not cases 13/14. Those require external storage/recomputation and
a genuinely streaming attention/layer schedule.

Case 2 is physically verified through four boards. Remaining work is
recovering the tiled schedule's compute overhead, scaling the same TCP runner to eight boards, and
designing the separate long-sequence storage strategy.
