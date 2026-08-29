# ESP32-C3 device test (XIAO ESP32-C3)

Firmware: TinyStories baseline, 4 layers, D=128, S=128, 4 heads.
Weights are embedded in flash (weights.bin fp32, weights_q12.bin Q12).

## Prerequisites
- PlatformIO (`pio`) 6.x, env `esp32-baseline`.
- The XIAO ESP32-C3 attached via USB (native USB-CDC).
- Device empty/held in bootloader? Normal `pio upload` handles it.

## Build + flash
    pio run -e esp32-baseline              # compile + link
    pio run -e esp32-baseline -t upload    # flash the board

Output sanity checks after flash:
    RAM: 81.7% (used 267804 bytes from 327680 bytes)  # .bss ~260.5 KB
    Flash: 83.3% (used 2621686 bytes from 3145728 bytes)

## Find the serial port
    pio device list          # look for a usbmodem* / usbserial* entry
  Typical: /dev/cu.usbmodem2101 (macOS).

## Run the end-to-end check
Run from the project root so `testdata/` resolves:

    python3 tools/device_test.py /dev/cu.usbmodem2101 --root . --seeds 0 1 2 3 4

The driver:
- captures the boot banner: `TM XIAO-ESP32C3 case2 baseline ready`
- `TM weights f32=1594368 bytes q12=786624 bytes` (blob embedding check)
- queries `M` / `S`; pads/recovers if a previous run left the firmware
  stuck in `read_input()` (no firmware timeout: an interrupted 'R' frame
  blocks the loop until a full 64 KB frame is delivered).
- For each seed: paces the 64 KB input (1 KB / 20 ms to avoid the native
  USB CDC RX drop), runs one forward, streams the 64 KB output back,
  parses `END forward=<n> us=<us>`, and compares against
  `testdata/ref_<seed>.bin` with the same gate as host_test
  (|a-b| <= 0.002 OR |a-b| <= 0.02*|b|).
- Timing sweep: `T <reps>` -> warmup 1 + `reps` forwards, prints per-forward us.

## Success criteria
- All seeds print `PASS` and the summary says all seeds passed.
- `END forward=<n> us=<us>` prints a finite, monotonically increasing `us`.
- Timing line prints `TM <mode> <us> ...` for each timed forward.

## Notes
- Firmware default mode is FAST (Q12 weights + Q15 activations); the
  int16 per-head attention quantization is included in both modes.
- The firmware serial loop has NO timeout in `read_input()`; if a test is
  interrupted, the driver's `wait_idle()`/`_kick()` recovers by completing
  the pending frame with zeros.
