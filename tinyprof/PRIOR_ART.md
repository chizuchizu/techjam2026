# tinyprof — prior art and what is actually claimed

`docs/PRIOR_ART.md` sets the standard for this repository: no world-first
framing, and a narrow claim that survives review. The same discipline applies to
the tooling.

## What already exists

tinyprof invents no measurement primitive. Everything it reads is standard:

- **ESP-IDF** provides `esp_timer_get_time()`, `esp_cpu_get_ccount()` (the
  RISC-V cycle counter), `heap_caps_*` allocation tracing, and
  `esp_get_minimum_free_heap_size()`.
- **FreeRTOS** provides `uxTaskGetStackHighWaterMark()`.
- **`idf.py size-components`** and the GNU binutils `size`/`nm` already report
  static section and symbol sizes from a linked ELF.
- **SEGGER SystemView**, **Percepio Tracealyzer** and **ARM Streamline** are
  mature embedded profilers with RTOS-aware trace, timeline views and far more
  capability than this.
- **`perf`**, **VTune** and **gprof** cover the host case comprehensively.

Anyone claiming to have invented profiling on a microcontroller is wrong, and
this file exists so that claim is never made on this project's behalf.

## What tinyprof actually claims

> tinyprof is a reproducible, dependency-free profiling harness that ties one
> physical ESP32-C3 capture to a single auditable artifact containing per-op
> time, call counts, derived memory traffic, ELF-derived static memory, runtime
> heap and stack watermarks, measured instrumentation overhead, and a
> roofline/MFU score — and renders a baseline-versus-optimised comparison from
> it. The contribution is the linkage and the auditability of the derivation,
> not the measurement primitives.

Three things follow from that phrasing, and they are the parts worth defending:

1. **The derivations are in the artifact.** The flash-traffic figure is not a
   number, it is a declared bytes-per-call model multiplied by a *measured* call
   count, with the reasoning attached to each op and a validation flag that goes
   false if the two disagree. It reproduces the previously hand-computed
   768 KiB/forward from first principles — and identifies 9,216 B of LayerNorm
   parameter reads the hand calculation had omitted.
2. **Two independent sources are cross-checked.** The firmware's own arena
   census is checked against the ELF's `.bss`. That check is not decorative: it
   is what found the 32 KB `a16` scratch buffer missing from the census.
3. **The tool measures its own distortion.** Probe cost is timed on the device at
   dump time and reported per build, because an overhead that differs between two
   firmwares is exactly what would inflate a speedup if it were folded in
   silently. When it exceeds 5% of the forward the artifact says the numbers
   should not be quoted, and the zone granularity is a build flag so the reader
   can act on that.

## What it is not

Not a tracer — there is no timeline and no event log. Not a sampler — there is no
PC sampling and no call-graph reconstruction. Not automatic — zones are placed by
hand, and a zone in the wrong place is a wrong number that no amount of tooling
will catch. Not general — the op vocabulary is frozen to this transformer's
structure, which is what makes two firmware revisions joinable at all.

Saying this plainly is what makes the rest of the claim survive a follow-up
question.
