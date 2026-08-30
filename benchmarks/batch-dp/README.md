# Batch cases 1, 3, 4, 5 — data-parallel across N boards

Cases 1, 3, 4 and 5 are the case-2 geometry at a larger batch:

| Case | Shape `(B,S,D,H,F,L)` | Batch |
|---:|---|---:|
| [3](../case-03/) | `(4,128,128,4,128,4)` | 4 |
| [4](../case-04/) | `(16,128,128,4,128,4)` | 16 |
| [1](../case-01/) | `(64,128,128,4,128,4)` | 64 |
| [5](../case-05/) | `(128,128,128,4,128,4)` | 128 |

Every one of the B inputs is an **independent forward over the same weights**,
so the right decomposition is data parallelism, not tensor parallelism: input
`i` runs on board `i % N` and the boards exchange **nothing**. Case 2 (`B=1`)
is the only one of the five that has to split a single forward, which is what
[`../case-02/multiboard/esp32-cluster-full/`](../case-02/multiboard/esp32-cluster-full/)
does.

Two consequences make this cheap:

* **One maintained model, two transports.** USB nodes run the default
  single-board build from
  [`../case-02/optimisation/esp32-baseline`](../case-02/optimisation/esp32-baseline).
  WiFi nodes use its opt-in `esp32-wifi-tiled` environment; the default image
  and published results remain unchanged.
* **No new weights.** The reference model is constructed from `(D, H, F, L)`
  under a fixed seed before any input is drawn, so the weights are byte-identical
  across all five cases. The boards keep case-2's `weights.bin` /
  `weights_q12.bin`. `tools/export_batch.py` emits only the per-case inputs and
  references, and the C forward reproduces them against those same blobs.

## Why this directory is shared

The four cases differ by one integer. Rather than duplicate the coordinator
four times, the implementation lives here and each case keeps its own results
and report under `case-0N/multiboard/results/`.

## Measurement convention

| Figure | Meaning |
|---|---|
| per-forward | one board's device time for one input |
| one board | what a single board would take for all B inputs |
| cluster | max over boards of its summed device forward times |
| speedup | one board / cluster |
| end-to-end | measured wall clock including the 64 KB in / 64 KB out per input over USB CDC or WiFi TCP |

The speedup uses device forward time on both sides, which is the same
convention the single-board and case-2 cluster numbers use — host serial
transfer is excluded from both. It is reported separately as *end-to-end* so
the transport cost is visible rather than hidden.

## Scaling beyond the batch

Data parallelism can only use `min(B, N)` boards, so case 3 (`B=4`) saturates
at four. Filling eight boards there means composing the two decompositions:
4-way data parallel across inputs, each input split 2-way by token row with the
case-2 cluster. Cases 4, 1 and 5 have enough inputs to keep 8 boards busy on
data parallelism alone.

## Reproduce

```bash
python3 tools/export_batch.py --batch 4 16 64 128     # inputs + references
../case-02/multiboard/esp32-cluster-full/tools/attach_boards.sh
# flash the single-board firmware to every board:
cd ../case-02/optimisation/esp32-baseline && pio run -t upload --upload-port <port>
cd -
python3 tools/run_batch_dp.py --batch 4 16 64 128
```

`testdata/` is generated, not committed (B=128 alone is 16 MB).

### WiFi nodes

The 16-row tiled FAST forward leaves enough DRAM for the Arduino WiFi/lwIP
stack. Each node exposes the same `M`/`R` protocol on a persistent TCP server
at port 5000, so the accuracy gate, retry handling, incomplete-run rule, and
speedup calculation are shared with the USB path.

```bash
cd ../case-02/optimisation/esp32-baseline
cp secrets.example.h secrets.h       # replace with the benchmark-LAN values
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM0

# Repeat the upload for each node, note the IP printed over USB, then:
cd ../../../batch-dp
python3 tools/run_batch_dp.py --batch 4 16 64 128 \
  --wifi 192.168.1.41 192.168.1.42
```

`HOST:PORT` is accepted when port 5000 is unavailable. `secrets.h` is ignored
by Git. A build without it remains usable over USB but intentionally does not
start WiFi, preventing placeholder credentials from being deployed silently.

Measured on two physical C3s for B=4: 4/4 forwards passed with no losses,
`2.00x` compute scaling, 4.218 s median forward, and 9.9 s end-to-end over
WiFi TCP. See
[`results_two_c3_wifi_tiled.json`](results_two_c3_wifi_tiled.json).

Measured on four physical C3s for B=4: one input per board, 4/4 forwards
passed, no losses, **4.00x** compute scaling, 4.215 s compute wall, and 6.5 s
end-to-end. See
[`results_four_c3_wifi_tiled.json`](results_four_c3_wifi_tiled.json).

The complete four-node sweep also passed cases 1, 3, 4 and 5: all 212/212
forwards passed with no losses, and compute scaling remained exactly 4.00x for
B=4, 16, 64 and 128. See
[`RESULTS_FOUR_C3_WIFI.md`](RESULTS_FOUR_C3_WIFI.md) and
[`results_cases_1_3_4_5_four_c3_wifi.json`](results_cases_1_3_4_5_four_c3_wifi.json).

Measured with eight physical C3s available, cases 1, 4 and 5 used all eight
nodes and retained **8.00x** compute scaling. Case 3 correctly saturated at
four active nodes and case 2 at one, because data parallelism cannot use more
workers than independent inputs. All 213/213 forwards across cases 1–5 passed
with no losses and zero failing elements. See
[`RESULTS_EIGHT_C3_WIFI.md`](RESULTS_EIGHT_C3_WIFI.md) for the complete timing
table, utilization limits, and raw-result links.
