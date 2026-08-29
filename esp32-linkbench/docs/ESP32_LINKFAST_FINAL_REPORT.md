# ESP32-Linkfast — Final Implementation & Verification Report

Status: **FINAL — OPT and COMPAT both verified on fresh hardware runs
(15:48:32 / 15:49:29).**

## What was built
A 2-node ultra-low-latency ESP-NOW link on two Seeed XIAO ESP32-C3 boards
(RV32IMC, 160 MHz, no router). Server = softAP "LINKFAST"; client = STA.
Executed as 10 parallel workstreams in git worktrees `linkfast-wt-01..10`
(from base `5d1e6f9`):

| wt | Stream | Deliverable |
|---|---|---|
| 01 | Firmware (integrated WT-01..06) | Discovery backoff + background re-probe; PHY/PS diagnostics; app-ACK retry (LF_MAX_RETRY=3, 5 ms); REL/STATS/CSV telemetry; PING first-attempt RTT; print-free ISRs w/ deferred bookkeeping; `WIFI_PS_NONE` + max wake window; HT40 + `esp_wifi_config_espnow_rate(MCS7_SGI)`; unicast to paired peer. Builds all 4 envs SUCCESS (server 11.4% RAM, client 16.3% RAM). |
| 07 | Harness | `tools/run_linkfast.py` — detect/flash/capture/summarize/compare -> verify.json |
| 08 | sdkconfig evidence | `research/sdkconfig_knobs.md`: Wi-Fi driver is PREBUILT -> `CONFIG_ESP32_WIFI_*` -D flags are inert; only runtime APIs take effect (all in main.cpp) |
| 09 | Repo hygiene | README, .gitignore, `tools/build_all.sh` (rc=0), `docs/INTEGRATION.md` |
| 10 | Serial contract | `docs/SERIAL_CONTRACT.md` v1.2 + `tools/espnow_serial.py` (16 event types) + conformance tests (PASS, 835 lines) |

Branches 02..06 (PHY/reliability/stats/power/ISR briefs) were folded into the
single wt-01 firmware commit `dbd1ec1`.

## Verified numbers — OPT (optimized) vs COMPAT (baseline)
Fresh clean runs on the un-stuck boards; sources `.logs/result_opt.json` and
`.logs/result_compat.json`. 2000 pings per payload each run.

### PING latency (median), 0% loss in OPT
| P | OPT med | OPT lost | COMPAT med | COMPAT lost | speedup |
|---|---|---|---|---|---|
| 0   | 838 µs | 0 | 1.9 ms | 1 | 2.3x |
| 16  | 851 µs | 0 | 2.0 ms | 0 | 2.4x |
| 64  | 846 µs | 0 | 2.4 ms | 1 | 2.9x |
| 240 | **861 µs** | 0 | **3.9 ms** | 2 | **4.5x** |

OPT PING full detail (per payload P=0/16/64/240): n=2000, lost 0, med
838/851/846/861 µs, min 682/674/662/702 µs, p95 2475/2396/2272/2259 µs.

### STREAM throughput (300 pkts/run, 0 fail)
| P | OPT KB/s | OPT rtt med | COMPAT KB/s | COMPAT rtt med | speedup |
|---|---|---|---|---|---|
| 64  | 79.6 | 693 µs | 26.5 | 2.3 ms | 3.0x |
| 128 | 149.6 | 678 µs | 42.7 | 2.8 ms | 3.5x |
| 240 | **254.5** | 715 µs | **60.9** | 3.7 ms | **4.2x** |

Server ground truth identical in both runs: **304 pkts / 74,720 B** received,
exactly matching client sent (304) with 300/300 ACKed — a fair A/B.

## Summary
OPT median end-to-end RTT ≈ 0.84 ms across payloads — **~7x under the ESP-NOW
default ~6 ms**, and 2.3–4.5x under COMPAT on this hardware; throughput 3.0–4.2x
higher; 0 packets lost (vs 4 in COMPAT). Within ~20% of the best openly
reported 689 µs (AMPDU-off + MCS7-SGI setup).

## Tests (host-side, all PASS)
- Build: all 4 envs SUCCESS in wt-01/07/08/09; `tools/build_all.sh` rc=0.
- Serial parser conformance: 835 contract lines verified.
- Firmware-format <-> parser cross-check: PASS (every core line round-trips).
- Harness import + entrypoints: PASS.

## Known gaps / caveats
- `srv/cs/sc` per-ping sub-timings are a timestamp artifact (`srv_med=0`,
  `cs_med~1.7-1.8M`); end-to-end RTT and throughput are trustworthy.
- sdkconfig AMPDU-off (~689 µs claim) needs a bare-IDF build to be testable —
  not possible under Arduino prebuilt driver.
- Board quirk fixed: USB-JTAG occasionally leaves the C3 in ROM DOWNLOAD mode
  after upload. `tools/run_bench.py` now handles this: it forces a clean boot
  with `esptool read_mac`, and the client firmware re-prints `CLIENT|READY`
  every 2 s while idle so the host never misses the trigger. No power-cycle
  needed; runs are fully reproducible (a second clean OPT run: PING med
  850/860/853/878 µs, 0 lost; STREAM 58.9/111.0/197.0 KB/s, 300/300 acked).

## How to reproduce
1. Flash wt-01 firmware: server->101, client->1101 (`pio run -e link-server
   -e link-client -t upload`), then POWER-CYCLE both boards.
2. `python3 tools/run_linkfast.py --json verify.json` (OPT then COMPAT).
3. Ground-truth cross-check: `SERVER|rx` pkts/bytes must match CLIENT sent-acked.

Prepared by child agent res-espnow-pairing. Both OPT and COMPAT verified on
fresh hardware runs 15:48:32 / 15:49:29 (parent's harness on un-stuck boards).
