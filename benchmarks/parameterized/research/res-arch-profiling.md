# Research: Profiling a C Transformer Function (Host + ESP32-C3)

Scope: how to time/profile a C transformer kernel both on an ESP32-C3 target and on the
host, which tools exist on each side, timing methodology for integer-only (no-FPU) kernels,
and how other people benchmark MCU transformer / GEMM kernels. Plus 1-2 reusable open-source
ESP32 / MCU transformer C projects.

Method note: in this session the Firecrawl *search* and *developer index* backends returned no
results (self-hosted instance has no search provider and the developer endpoint is absent), so
this report was assembled with `firecrawl scrape` against primary documentation pages plus the
GitHub REST API for repository discovery. All URLs are listed at the bottom.

---

## 1. Target-side (ESP32-C3) timing primitives

### 1.1 `esp_timer_get_time()` — microsecond wall clock
Source: ESP-IDF "ESP Timer (High Resolution Timer)" docs.

- Returns `int64_t` microseconds since boot (ESP Timer init, just before `app_main`).
- Resolution **1 microsecond**, 52-bit range, 52-bit timer.
- Documented as "fast with **no locking**", usable in both tasks and **ISR** routines.
- Not affected by timezone/DST; resets to zero after deep sleep.

```c
#include "esp_timer.h"
int64_t start = esp_timer_get_time();
kernel();
int64_t end = esp_timer_get_time();
// (end - start) microseconds
```

ESP-IDF's own speed guide says `esp_timer_get_time()` gives microsecond-precision wall time
"but has moderate overhead each time the timing functions are called." The overhead is why,
for a very short kernel, you loop it many times and divide.

### 1.2 `esp_cpu_get_cycle_count()` — cycle counter (best for kernels)
Source: `esp_hw_support/include/esp_cpu.h` (ESP-IDF v6.1), confirmed in source.

- `esp_cpu_get_cycle_count()` returns the CPU cycle count (`esp_cpu_cycle_count_t`).
- On Xtensa it maps to `xt_utils_get_cycle_count()` (CCOUNT); on RISC-V (ESP32-C3) it maps
  to `rv_utils_get_cycle_count()`, i.e. the `cycle` CSR (`rdcycle`).
- This is the **lowest-overhead** portable timer; use it for microbenchmarks of routines that
  run in hundreds or thousands of cycles.
- `esp_cpu_get_ccount()` is the older Xtensa-specific name. ESP-IDF >= 5.0 code commonly does:

```c
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#define esp_cpu_get_ccount esp_cpu_get_cycle_count
#endif
```

- Convert cycles to time: ESP32-C3 max/default frequency is **160 MHz**, so
  `us = cycles / 160`. (Do not hard-code if dynamic frequency scaling / power management is
  active; reading the configured `esp_clk_cpu_freq()` / `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ`
  is safer.)

Canonical profiling pattern (copied verbatim from espressif/esp-nn `test_app/main/main.c`),
which is exactly the "profile a C kernel" idiom on ESP32:

```c
static uint32_t start_c, start_opt, total_c, total_opt;
void profile_c_start() { start_c = esp_cpu_get_ccount(); }
uint32_t profile_c_end() { total_c = esp_cpu_get_ccount() - start_c; return total_c; }
// ... same for profile_opt_start/end ...
printf("PROFILE: %s, ansi=%"PRIu32", opt=%"PRIu32", speedup=...", kernel, total_c, total_opt);
```

### 1.3 `esp_timer_dump()` + `CONFIG_ESP_TIMER_PROFILING`
If the function under test is reached from ESP Timer callbacks, enable
`CONFIG_ESP_TIMER_PROFILING`; `esp_timer_dump(stdout)` then prints per-timer columns
including `Cb_exec_time` (total microseconds spent inside each callback).

### 1.4 GPTimer for peripheral-accurate timing
For waveform-grade accuracy or event capture, ESP-IDF recommends GPTimer over ESP Timer
(resolution below tens of microseconds). Rarely needed for a transformer kernel, but GPTimer
can capture hardware events/alarms if you want an externally visible timer.

---

## 2. ESP-IDF performance tools & SEGGER SystemView

### 2.1 Espressif "Speed" guide — official measurement advice
Source: ESP-IDF API Guides > Performance > Speed.

The official recipe for measuring an important function:

```c
#include "esp_timer.h"
void measure_important_function(void) {
    const unsigned MEASUREMENTS = 5000;
    uint64_t start = esp_timer_get_time();
    for (int retries = 0; retries < MEASUREMENTS; retries++) {
        important_function();
    }
    uint64_t end = esp_timer_get_time();
    printf("%u iterations took %llu ms (%llu us per invocation)\n",
           MEASUREMENTS, (end - start)/1000, (end - start)/MEASUREMENTS);
}
```

Explicit guidance:
- `esp_timer_get_time()` = microsecond wall time, moderate call overhead.
- `gettimeofday()` / `utime()` = slightly higher overhead.
- `cpu_hal_get_cycle_count()` (legacy name) / `esp_cpu_get_cycle_count()` = **lowest
  overhead**, best for very short routines.
- Run the target many times to average out RTOS context switches and measurement overhead.
- Microbenchmarks shorter than ~1-2 ms can vary a lot because of **flash-cache misses whose
  pattern depends on binary layout**. Repeat many times, or move the hot code to **IRAM**.

### 2.2 Application Level Tracing + SystemView
Source: ESP-IDF API Guides > App Level Tracing; SEGGER SystemView product page.

- ESP-IDF has an Application Level Tracing (app_trace) library over **JTAG, UART, or USB**
  with small runtime overhead.
- The UART transport is the standard path to **SEGGER SystemView**.
- SystemView is a real-time recording/visualization tool: it shows **tasks, ISRs, software
  timers, scheduler events, and CPU load** with per-event timestamps.
- ESP-IDF SystemView support comes from the managed component **`espressif/esp_sysview`**:

```yaml
dependencies:
  espressif/esp_sysview: ^1
```

  then menuconfig: `Component config > ESP Trace Configuration > Trace library >
  External library from component registry`, and configure under
  `Component config > SEGGER SystemView Configuration`.

- On ESP32-C3, `CONFIG_ESP_TRACE_TIMESTAMP_SOURCE` selects the timestamp source: the internal
  **cycle counter at max CPU frequency** in single-core mode (external 1/2-frequency timer in
  dual-core mode).
- Per-event collection can be toggled: trace-buffer overflow, ISR enter/exit, ISR-exit-to-
  scheduler, task start/stop execution, task ready states, task create/terminate, idle, timer
  enter/exit.
- With OpenOCD you can stream traces: `esp sysview start file://out.svd ...`.

This is the right tool when the question is "who is preempting my kernel / where is the CPU
time going across the whole system", rather than "how many cycles does one function take".

---

## 3. FreeRTOS task statistics

Source: FreeRTOS "Run Time Statistics" page; ESP-IDF FreeRTOS API reference.

Run-time stats answer "which task is using CPU (%)" — useful to confirm the transformer task
actually gets the CPU, and to detect that your measurement task is being preempted.

Enable in `FreeRTOSConfig.h`:

```c
#define configGENERATE_RUN_TIME_STATS 1
#define portCONFIGURE_TIMER_FOR_RUN_TIME_STATS() /* configure a fast timer */
#define portGET_RUN_TIME_COUNTER_VALUE()         /* read that timer */
```

FreeRTOS guidance:
- The run-time time base must be **higher resolution than the tick interrupt**; recommended
  **10-100x faster** than the tick. Faster = more accurate, but overflows sooner.
- `portCONFIGURE_TIMER_FOR_RUN_TIME_STATS()` is called automatically by
  `vTaskStartScheduler()`.
- `vTaskGetRunTimeStats(buffer)` prints a table: `Task    Abs Time    % Time`.
- `uxTaskGetSystemState()` (needs `configUSE_TRACE_FACILITY=1`) fills a
  `TaskStatus_t` array with `ulRunTimeCounter` and total run-time; `vTaskGetRunTimeStats()`
  is just a human-readable wrapper over it. It suspends the scheduler while it runs, so it is
  a debugging API, not a per-kernel timer.
- Other useful FreeRTOS timestamps: `xTaskGetTickCount()` (tick-resolution, 1-10 ms at
  100-1000 Hz ticks).

On ESP-IDF the typical setup is to drive the run-time counter from the same cycle counter /
`esp_timer` used above.

---

## 4. Host-side profilers for C code

### 4.1 Linux `perf` (sampling + counters) — recommended
Source: Brendan Gregg's perf examples page (brendangregg.com/perf.html).

```bash
# aggregate hardware counters for one process/command
perf stat ./transformer

# sample on-CPU functions at 99 Hz
perf record -F 99 ./transformer
perf report                      # TUI: hot functions + call graph
perf annotate <symbol>           # hot assembly-level attribution

# call stacks with frame pointers / DWARF unwind
perf record -F 99 -g -- ./transformer
# build host code with: -g -fno-omit-frame-pointer  (or use dwarf unwinding)
```

Use `perf list`, `perf stat -d`, and `perf stat -e cycles,instructions,cache-references,
cache-misses` to separate CPU-bound vs cache-bound behavior.

### 4.2 Valgrind `callgrind` (deterministic instruction-level + call graph)
Source: Valgrind Callgrind manual.

```bash
valgrind --tool=callgrind ./transformer     # writes callgrind.out.<pid>
callgrind_annotate callgrind.out.<pid>      # sorted functions, source annotation
# GUI: KCachegrind
```

Callgrind records **instruction counts, caller/callee graph, and call counts**; optional cache
simulation and branch prediction. It is deterministic but ~10-50x slower than native — ideal
for a small C kernel/unit test where reproducibility matters more than wall time.

### 4.3 GNU `gprof` (instrumented flat + call-graph profile)
Source: GNU gprof manual (binutils 2.47).

```bash
gcc -pg -g -O2 -o transformer src.c
./transformer                       # produces gmon.out
gprof ./transformer gmon.out        # flat profile + call graph
```

`-pg` instruments function entry/exit; gives flat profile (time/self per function) and a call
graph. Statistical sampling error and children-time estimation caveats apply (documented in
the manual). Easy but compiler-dependent; prefer `perf`/callgrind when available.

### 4.4 macOS `Instruments` — Time Profiler
- `Instruments` is the GUI profiler shipped with Xcode; the **Time Profiler** instrument is a
  low-overhead sampling profiler for CPU usage.
- Run the target in Xcode (Product > Profile, or Instruments.app, then choose "Time Profiler"),
  record, and inspect per-function CPU time and call trees.
- CL equivalent for quick checks: `xcrun xctrace record --template 'Time Profiler' --launch ./transformer`.
- Note: Apple's help pages and the `instruments` documentation require JavaScript, so primary
  facts here are from the tool itself; Apple developer pages on "Recording a Time Profile"
  document the workflow.

Suggested host workflow for this project (cross-checks the same C kernel):
1. `-O2 -g -fno-omit-frame-pointer` build.
2. `perf stat` for cycles/instructions/CPI and cache misses.
3. `perf record -F 99 -g` + `perf report`/`perf annotate` for hotspot attribution.
4. `callgrind` on a small deterministic unit test for exact instruction counts (useful when
   comparing two kernel variants, e.g. scalar vs fixed-point).
5. Mirror the exact same benchmark loop on the MCU (next section) so host and target numbers
   are comparable.

---

## 5. Timing methodology for no-FPU integer kernels (ESP32-C3)

ESP32-C3 is a single-core RV32IMC **without an FPU** (no F/D extensions), so any transformer
kernel runs as fixed-point / integer math, often `q15` or integer residual arithmetic. Timing
methodology:

1. **Use the cycle counter, not the wall clock.** `esp_cpu_get_cycle_count()` (rdcycle) is
   the lowest-overhead and most precise; at 160 MHz it quantizes to 6.25 ns. Use
   `esp_timer_get_time()` only for coarse (>~100 us) or end-to-end measurements.

2. **Loop the kernel N times and divide.** A single short call is dominated by the two
   counter reads and by cache state. ESP-IDF's own guidance: 5000 iterations, subtract the
   loop overhead (time an empty loop) or choose N big enough that timer-call overhead is
   negligible, then divide by N.

3. **Pin down the memory/cache contribution.** Integer kernels are often memory-bound at low
   precision. On ESP32-C3, code and data can live in flash (XIP, cached), IRAM, or DRAM.
   - Microbenchmarks < 1-2 ms vary with **flash cache misses that depend on binary layout**.
   - Put the hot kernel in **IRAM** (`IRAM_ATTR`) and weights in DRAM to measure the compute
     bound; also measure from flash to see the real deployment cost.
   - Report both numbers and the exact `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ` / PSRAM config.

4. **Average out RTOS noise.** Run the loop inside one task, with interrupts enabled, and
   take min/median/mean over several trials; or (for cycle-exact single runs) place the
   kernel in a critical section / IRAM and briefly disable preemption to see the cycle count
   without scheduler interference.

5. **Convert cycles to time at the configured clock.** `cycles / 160e6` = seconds at 160 MHz.
   Watch out for dynamic frequency scaling (DFS); if the kernel runs long enough for the
   governor to downclock, pin the CPU to max or read the actual frequency.

6. **Separate fixed-point conversion cost from the matmul cost.** Instrument the layer
   boundaries (QKV projection, attention softmax/q-alias, linear head), not just the whole
   transformer. That is exactly how esp-nn and PULP transformer kernels are benchmarked
   (per-kernel cycle counts).

7. **Validate with a GPIO toggle / logic analyzer for inductive checking.** Toggle a pin
   before/after the kernel and measure with a scope; this confirms the in-code timer is not
   lying and gives an independent end-to-end figure (also works if timing is buried in an
   ISR or DMA flow).

---

## 6. How others profile MCU transformer / GEMM kernels

- **espressif/esp-nn** (the ESP-IDF NN/GEMM library) reports per-kernel performance in
  **"ticks taken for kernel to execute"** — i.e. raw CPU cycle counts — and compares
  `ANSI C` vs `Optimized` with an "Opt Ratio". Its `test_app/main/main.c` measures each
  kernel with `esp_cpu_get_ccount()` start/end wrappers exactly as shown in section 1.2.
  Source: https://github.com/espressif/esp-nn — README "Performance".
- **PULP/pulp-transformer** ("Optimizing the Deployment of Tiny Transformers on Low-Power
  MCUs", arXiv:2404.02945) ships a **kernel test harness** (`kernelTest.sh` / `testConfig.yaml`)
  that runs individual attention kernels (e.g. `projQK`, 4x2 linear projection) against
  "golden" reference kernels and reports **latency (ms) and energy (uJ)** per layer on
  STM32H7/L4 and GAP9. This is the reference pattern for MCU transformer kernel benchmarking:
  per-layer, golden-model checked, latency + energy, not just a single end-to-end number.
- **CMSIS-NN / PULP-NN-style benchmarks** measure per-kernel cycles on Cortex-M
  (CMSIS-NN uses the ARM **DWT->CYCCNT** counter) and per-layer MAC counts, MACC/s, or ms.
  The cycle counter on ARM Cortex-M is `DWT_CYCCNT`; on RISC-V it is `rdcycle`/`mcycle` —
  the direct analogue of what ESP32-C3 uses.
- **Application-level throughput** is how end-to-end transformer runs are reported on MCUs:
  `DaveBben/esp32-llm` reports **19.13 tok/s** for a 260K-parameter llama2.c transformer on
  an ESP32-S3 (dual-core math, esp-dsp dotprod, 240 MHz, 2 MB PSRAM). Throughput alone hides
  per-layer splits, so combine it with per-kernel cycle counts.
- **System-level tracing** (SEGGER SystemView over UART) is the standard way to attribute
  time across tasks/ISRs and confirm the inference task is getting the expected CPU share.

---

## 7. Open-source ESP32 / MCU transformer C projects (with links)

### 7.1 `DaveBben/esp32-llm` — ESP32-S3 transformer inference in C
- https://github.com/DaveBben/esp32-llm
- Language: **C**; stars ~611.
- Runs a 260K-parameter **tinyllamas (llama2.c) transformer** fully on-device on an
  ESP32-S3FH4R2 (2 MB PSRAM), **19.13 tok/s**.
- Optimizations: dual-core math, ESP-DSP SIMD dot-product functions, 240 MHz CPU + 80 MHz
  PSRAM, larger instruction cache. Forked/derived from `karpathy/llama2.c`.
- Directly relevant as a small, readable "transformer function on ESP32" reference.

### 7.2 `pulp-platform/pulp-transformer` — tiny transformer kernel library for MCUs
- https://github.com/pulp-platform/pulp-transformer
- Language: **C**; paper arXiv:2404.02945.
- "Optimizing the Deployment of Tiny Transformers on Low-Power MCUs": optimized attention/
  linear kernels (fused-weight MHSA "FWSA", depth-first tiling "DFT" to cut the attention-map
  memory peak), evaluated on STM32H7, STM32L4, and GAP9 (RISC-V). Includes the kernel test
  harness described in section 6. Reported: 4.79x faster than CMSIS-NN on ARM and 2.0x vs
  PULP-NN on RISC-V; a GAP9 transformer block at 0.14 ms / 4.92 uJ.
- Not ESP32-specific, but the closest thing to a reusable, benchmarked **MCU transformer C
  kernel library** — its per-kernel profiling pattern ports directly to ESP32-C3.

Honorable mentions (less directly reusable):
- `ahmedbarakat207/espllm` — "a Transformer Based Model for ESP32 (520KB Memory)", C++.
- `Carloscodix/qapla` — char-level transformer trained/inferred on an $8 ESP32-S3, C++.
- `imFARSI/NanoMind-S3` — dual-core INT4 15.2M-parameter transformer on ESP32-S3, C.
- `fkkarakurt/Nerve` — single-header pure-C LLM/embeddings engine (portable MCU target), C.

---

## 8. Recommendations for this project

1. **On target (ESP32-C3):** wrap the transformer function with `esp_cpu_get_cycle_count()`
   in an N-iteration loop (the esp-nn pattern), report **cycles/call**, then convert with the
   real CPU MHz. Also print `esp_timer_get_time()` delta for end-to-end sanity.
2. **On host:** build with `-O2 -g -fno-omit-frame-pointer`; run `perf stat` (cycles,
   instructions, IPC, cache misses), `perf record -F 99 -g` + `perf report`/`annotate`;
   use `callgrind` + KCachegrind on a deterministic unit test to compare kernel variants.
3. **Context:** enable FreeRTOS run-time stats on the C3 (`configGENERATE_RUN_TIME_STATS`
   plus the two run-time-counter macros; in ESP-IDF menuconfig this is the FreeRTOS
   run-time-stats option) and log `vTaskGetRunTimeStats()` once at the end to confirm the
   compute loop is not starved; add
   SystemView (`espressif/esp_sysview` over UART) only if you need whole-system timing
   evidence.
4. **Memory-bound vs compute-bound:** time the kernel from flash and from IRAM separately;
   integer/no-FPU kernels often show this split more than the arithmetic itself.
5. **Benchmark against reference projects:** reuse the esp-nn per-kernel style and, if the
   workload matches, compare numbers against `pulp-transformer` and `esp32-llm` reports.

---

## 9. Sources

ESP32 / ESP-IDF:
- ESP Timer (High Resolution Timer): https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/system/esp_timer.html
- Performance > Speed: https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-guides/performance/speed.html
- App Level Tracing / SystemView: https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-guides/app_trace.html
- FreeRTOS (ESP-IDF) API: https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/system/freertos_idf.html
- esp_cpu.h source (cycle count): https://github.com/espressif/esp-idf/blob/v6.1/components/esp_hw_support/include/esp_cpu.h
- esp-nn profile harness: https://github.com/espressif/esp-nn/blob/master/test_app/main/main.c and README https://github.com/espressif/esp-nn

FreeRTOS / SEGGER:
- FreeRTOS Run Time Stats: https://www.freertos.org/rtos-run-time-stats.html
- SEGGER SystemView: https://www.segger.com/products/development-tools/systemview/

Host profilers:
- Brendan Gregg perf examples: https://www.brendangregg.com/perf.html
- Valgrind Callgrind manual: https://valgrind.org/docs/manual/cl-manual.html
- GNU gprof manual: https://sourceware.org/binutils/docs/gprof/
- Apple Xcode "Recording a Time Profile": https://developer.apple.com/documentation/xcode/recording-a-time-profile

MCU transformer projects:
- DaveBben/esp32-llm: https://github.com/DaveBben/esp32-llm
- pulp-platform/pulp-transformer: https://github.com/pulp-platform/pulp-transformer (paper https://arxiv.org/abs/2404.02945)
