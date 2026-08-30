/*
 * tinyprof.h - operator-level profiler for the case-2 transformer on ESP32-C3.
 *
 * Design constraints that shape this file:
 *   * The optimised build has ~384 B of free DRAM (optimisations/20_...md).
 *     Nothing here may add static RAM. Every buffer used while dumping lives
 *     on the caller's stack; every optional metric is behind a build flag that
 *     is off by default; the arena census lives in flash via weak functions.
 *   * The same sources must build on the host, so tools/host_test.c keeps
 *     working. TP_NOW() falls back to clock_gettime(CLOCK_MONOTONIC).
 *   * The slot vocabulary is frozen: it is the JSON key set and the join key
 *     between the baseline and the optimised capture. Adding a slot is fine;
 *     renaming or reordering one breaks every committed artifact.
 *
 * This replaces the hand-rolled TM_PROFILE block that used to live in model.c;
 * PB/PE/PBT/PET remain as aliases so no existing call site moves.
 */
#ifndef TINYPROF_H
#define TINYPROF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TINYPROF defaults on unless the translation unit opted out (the UART worker
 * env sets -DTM_NO_MODEL_PROFILE=1 to keep its forward uninstrumented). */
#ifndef TINYPROF
#  ifdef TM_NO_MODEL_PROFILE
#    define TINYPROF 0
#  else
#    define TINYPROF 1
#  endif
#endif

/* Frozen slot vocabulary. Order defines the wire index `i`. */
enum {
    TP_NORM1 = 0, TP_QKV, TP_QUANT, TP_ATTN, TP_OPROJ, TP_RES1,
    TP_NORM2, TP_F1, TP_GELU, TP_F2, TP_RES2, TP_FINAL,
    TP_ATTN_QK, TP_ATTN_EXP, TP_ATTN_PV,
    TP_SLOTS
};

/* Shape + clock of the build, echoed in the header line so an artifact is
 * self-describing without the host having to know which case it came from. */
typedef struct {
    int S, D, H, F, L;
    int mhz;
} tp_shape_t;

/* Arena census entry. Tables are `static const` in the model translation unit,
 * so they live in flash .rodata and cost no DRAM. */
typedef struct {
    const char* name;   /* symbol name, e.g. "g_buf1" */
    uint32_t    bytes;
    const char* kind;   /* e.g. "fp32_activation", "q15_head", "scratch" */
} tp_arena_t;

/* ---- hooks the firmware/model overrides (all weak, all flash-resident) ---- */

/* One line of profiler output. Device main.cpp overrides with Serial.print. */
void tp_emit(const char* line);

/* Arena census. The model TU defines these; the defaults report nothing. */
const tp_arena_t* tp_arena_table(void);
int               tp_arena_count(void);

/* Build shape. The model TU defines this; the default reports zeros. */
const tp_shape_t* tp_shape(void);

/* Zone nesting: parent slot index per slot, or -1 for a direct child of the
 * forward. Defaulted for the case-2 layout; override if zones nest elsewhere. */
const signed char* tp_parents(void);

/* Platform metrics (heap, stack, clock). Real on ESP32, no-op on host. */
void tp_dump_platform(void);

/* Build identity, echoed in the header line so an artifact is self-describing.
 * Overridden per firmware; defaults to "unknown". */
const char* tp_build_tag(void);

/* ---- API ---- */

void tp_reset(void);                 /* zero all slots and the wall clock */
void tp_dump(void);                  /* emit the full TPROF|... record set */

/* Two clocks, deliberately. tp_now_us() is the 64-bit wall clock used for the
 * whole forward (42 s on the baseline, so it must not wrap). tp_ticks() is the
 * free-running high-resolution counter used for zones: the RISC-V cycle counter
 * on the C3 (6.25 ns at 160 MHz), monotonic nanoseconds on the host. It is
 * 32-bit and wraps; deltas are taken in uint32 so the wrap is harmless, which
 * bounds a single zone at ~26.8 s on device. tick_hz travels in the wire header
 * so the host never has to assume a unit. */
uint64_t tp_now_us(void);
uint32_t tp_ticks(void);
uint32_t tp_tick_hz(void);

/* CPU clock in Hz, used to convert cycles to time. Override per board. */
#ifndef TP_CPU_HZ
#define TP_CPU_HZ 160000000u
#endif

#if TINYPROF

void tp_begin(int slot);
void tp_end(int slot);
void tp_wall_begin(void);
void tp_wall_end(void);

#define TP_B(i) tp_begin(i)
#define TP_E(i) tp_end(i)

#else

#define TP_B(i) do {} while (0)
#define TP_E(i) do {} while (0)
#define tp_wall_begin() do {} while (0)
#define tp_wall_end()   do {} while (0)

#endif /* TINYPROF */

/* Back-compat aliases: model.c's existing call sites are PB/PE/PBT/PET. */
#define PB(i) TP_B(i)
#define PE(i) TP_E(i)
#define PBT() tp_wall_begin()
#define PET() tp_wall_end()

/* Note: tm_profile_dump() is deliberately NOT macro-aliased here. model.c and
 * model_tiled.c each keep their own one-line definition that calls tp_dump(),
 * so the two translation units do not collide over a single symbol. */

#ifdef __cplusplus
}
#endif

#endif /* TINYPROF_H */
