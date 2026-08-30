/*
 * tinyprof.c - portable core: slot storage, overhead calibration, wire dump.
 *
 * Static RAM budget (the optimised build has ~384 B of free DRAM):
 *   p_start[15] int64 = 120 B      } these three already existed inside
 *   p_acc[15]   uint64 = 120 B     } model.c's TM_PROFILE block, so moving
 *   p_cnt[15]   uint32 =  60 B     } them here is a net-zero .bss change.
 *   p_wall      uint64 =   8 B
 *   ------------------------------
 *   308 B, unchanged from the block this replaces. The 160-byte `line`
 *   buffer model.c used to hold statically is now a stack local in tp_dump(),
 *   so the swap is net -160 B.
 *
 * Everything else - arena census, build tag, emit hook - is a weak function,
 * so it lives in flash and costs no DRAM at all.
 */
#include "tinyprof.h"

#include <stdio.h>
#include <string.h>

/* ---- backend: two clocks ---------------------------------------------
 *
 * Zones are timed with a free-running TICK counter, not the microsecond timer.
 * The reason is a measurement bug this replaces: with a 1 us timer, every zone
 * shorter than a microsecond rounds to zero. In the original block those zones
 * were then dropped by an `if (d > 0)` guard, so `res1` disappeared from the
 * profile entirely and `res2` reported 2 calls out of 12 - a silent undercount
 * of both time and call count, in exactly the cheap ops that an optimisation
 * pass is trying to drive toward zero.
 *
 * On the C3 the tick source is the RISC-V cycle counter: 6.25 ns resolution at
 * 160 MHz, and a cheaper read than esp_timer_get_time() since it is a CSR read
 * rather than a call into the timer driver. It is 32-bit and wraps every ~26.8 s,
 * which is fine for a zone (the longest single zone in the 42 s baseline is one
 * `attn` call at ~1.9 s) but not for the whole forward - so the per-forward wall
 * keeps using the 64-bit microsecond timer. Deltas are taken in uint32 so the
 * wrap is handled by modular arithmetic and never needs a special case.
 *
 * The host build uses monotonic nanoseconds truncated the same way, so the same
 * code path is exercised natively. tick_hz travels in the header line and the
 * analyzer does the conversion, so no unit is ever assumed host-side. */

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
#include <esp_timer.h>
#include <esp_cpu.h>
#include <esp_idf_version.h>

/* IDF renamed the cycle-counter helper in v5. Arduino-ESP32 here pins IDF 4.4,
 * which spells it esp_cpu_get_ccount(); kernels.c already depends on that name.
 * Support both so the profiler is not the thing that breaks on an IDF bump. */
#if ESP_IDF_VERSION_MAJOR >= 5
#  define TP_RAW_CYCLES() esp_cpu_get_cycle_count()
#else
#  define TP_RAW_CYCLES() esp_cpu_get_ccount()
#endif

uint64_t tp_now_us(void)  { return (uint64_t)esp_timer_get_time(); }
uint32_t tp_ticks(void)   { return (uint32_t)TP_RAW_CYCLES(); }
uint32_t tp_tick_hz(void) { return (uint32_t)(TP_CPU_HZ); }
#else
#include <time.h>
uint64_t tp_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)(ts.tv_nsec / 1000);
}
uint32_t tp_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}
uint32_t tp_tick_hz(void) { return 1000000000u; }
#endif

/* ---- weak hooks ------------------------------------------------------ */

__attribute__((weak)) void tp_emit(const char* line) { (void)line; }
__attribute__((weak)) const tp_arena_t* tp_arena_table(void) { return 0; }
__attribute__((weak)) int tp_arena_count(void) { return 0; }
__attribute__((weak)) const char* tp_build_tag(void) { return "unknown"; }

static const tp_shape_t tp_shape_unknown = { 0, 0, 0, 0, 0, 0 };
__attribute__((weak)) const tp_shape_t* tp_shape(void) { return &tp_shape_unknown; }

/* No-op on the host build; tinyprof_esp32.c provides the real one. */
__attribute__((weak)) void tp_dump_platform(void) { }

/* kernels.c defines these when its own TM_PROFILE block is compiled in.
 * Weak stubs keep tinyprof linkable in a project that has no kernels.c. */
__attribute__((weak)) void tm_kbench_clear(void) { }
__attribute__((weak)) void tm_kbench_dump(void) { }

/* ---- slot state ------------------------------------------------------ */

static const char* const tp_name[TP_SLOTS] = {
    "norm1", "qkv", "quant", "attn", "oproj", "res1",
    "norm2", "f1", "gelu", "f2", "res2", "final",
    "attn_qk", "attn_exp", "attn_pv"
};

/* Zone nesting. Several zones are measured *inside* another, so their time is
 * already included in the parent's total:
 *
 *   quant   inside qkv   (the per-head Q15 projection GEMM sits inside the
 *                         qkv bracket - model.c phase A and phase B both)
 *   gelu    inside f2    (the activation is bracketed inside the second FFN
 *                         GEMM's zone)
 *   attn_qk / attn_exp / attn_pv  inside attn
 *
 * Getting this wrong is the classic way a profiler report lies: summing
 * inclusive times gives >100% of the forward, and a "top 10 by time" ranking
 * double-counts the parent. The host analyzer subtracts children to get
 * exclusive time and ranks on that, and it can only do so because the firmware
 * states the tree rather than leaving the host to guess from the names.
 *
 * -1 means the zone hangs directly off the forward. Weak, so a firmware whose
 * zones nest differently can override it without touching this file. */
static const signed char tp_parent_default[TP_SLOTS] = {
    /* norm1 */ -1, /* qkv   */ -1, /* quant */ TP_QKV,  /* attn  */ -1,
    /* oproj */ -1, /* res1  */ -1, /* norm2 */ -1,      /* f1    */ -1,
    /* gelu  */ TP_F2, /* f2  */ -1, /* res2 */ -1,      /* final */ -1,
    /* attn_qk */ TP_ATTN, /* attn_exp */ TP_ATTN, /* attn_pv */ TP_ATTN
};

__attribute__((weak)) const signed char* tp_parents(void) { return tp_parent_default; }

#if TINYPROF

/* 240 B of .bss, against the 308 B the block this replaces used - so wiring
 * tinyprof into the optimised build frees RAM rather than consuming it, which
 * matters because that build has ~384 B of DRAM left. */
static uint32_t p_start[TP_SLOTS];
static uint64_t p_acc[TP_SLOTS];
static uint32_t p_cnt[TP_SLOTS];
static uint64_t p_wall_us;

void tp_begin(int slot) { p_start[slot] = tp_ticks(); }

/* The count is incremented unconditionally. A zone that measures zero ticks is
 * still a zone that ran, and dropping it is how the previous implementation
 * lost `res1` completely. Zero-tick zones are instead surfaced host-side as
 * resolution-limited, which is a statement the reader can check. */
void tp_end(int slot) {
    uint32_t d = tp_ticks() - p_start[slot];   /* modular: wrap-safe */
    p_acc[slot] += (uint64_t)d;
    p_cnt[slot]++;
}

void tp_wall_begin(void) { p_wall_us = tp_now_us(); }
void tp_wall_end(void)   { p_wall_us = tp_now_us() - p_wall_us; }

void tp_reset(void) {
    for (int i = 0; i < TP_SLOTS; i++) { p_acc[i] = 0; p_cnt[i] = 0; p_start[i] = 0; }
    p_wall_us = 0;
    tm_kbench_clear();
}

/* Cost of one tp_ticks() read, in ticks, measured on the spot.
 *
 * Reported rather than silently subtracted: instrumentation costs two reads per
 * zone and the forward runs ~24k zones, which is noise against the 42 s
 * baseline but on the order of a percent against the 2.0 s optimised build. An
 * overhead that differs by build is exactly the kind of thing that quietly
 * inflates a speedup, so it travels in the artifact and the analyzer carries
 * both as-measured and overhead-corrected times, labelled apart.
 *
 * Stack locals only - no static state. */
static uint32_t tp_probe_ticks_x1000(void) {
    const int N = 4096;
    uint32_t t0 = tp_ticks();
    uint32_t sink = 0;
    for (int i = 0; i < N; i++) sink += tp_ticks();
    uint32_t t1 = tp_ticks();
    if (sink == 0xFFFFFFFFu) tp_emit("");   /* keep the loop, without a static */
    uint32_t d = t1 - t0;
    return (uint32_t)(((uint64_t)d * 1000ull) / (uint64_t)N);
}

void tp_dump(void) {
    char line[192];

    const tp_shape_t* sh = tp_shape();
    (void)snprintf(line, sizeof line,
        "TPROF|hdr|v=1|tag=%s|slots=%d|S=%d|D=%d|H=%d|F=%d|L=%d|mhz=%d|tick_hz=%lu\n",
        tp_build_tag(), (int)TP_SLOTS,
        sh->S, sh->D, sh->H, sh->F, sh->L, sh->mhz,
        (unsigned long)tp_tick_hz());
    tp_emit(line);

    uint64_t calls = 0;
    for (int i = 0; i < TP_SLOTS; i++) {
        if (!p_cnt[i]) continue;
        calls += p_cnt[i];
        (void)snprintf(line, sizeof line,
            "TPROF|op|i=%d|name=%s|ticks=%llu|n=%lu|parent=%d\n",
            i, tp_name[i], (unsigned long long)p_acc[i],
            (unsigned long)p_cnt[i], (int)tp_parents()[i]);
        tp_emit(line);
    }

    (void)snprintf(line, sizeof line, "TPROF|wall|us=%llu\n",
                   (unsigned long long)p_wall_us);
    tp_emit(line);

    uint32_t tk1000 = tp_probe_ticks_x1000();
    (void)snprintf(line, sizeof line,
        "TPROF|ovh|milliticks_per_probe=%lu|probes=%llu|ticks=%llu\n",
        (unsigned long)tk1000, (unsigned long long)(calls * 2),
        (unsigned long long)((calls * 2ull * (uint64_t)tk1000) / 1000ull));
    tp_emit(line);

    tp_dump_platform();          /* heap / stack / clock, or nothing on host */

    const tp_arena_t* a = tp_arena_table();
    int an = tp_arena_count();
    for (int i = 0; i < an; i++) {
        (void)snprintf(line, sizeof line,
            "TPROF|arena|name=%s|bytes=%lu|role=%s\n",
            a[i].name, (unsigned long)a[i].bytes, a[i].kind);
        tp_emit(line);
    }

    /* kernels.c's own microbench, in its established legacy format, so
     * FLASH_TEST.md and any teammate's parser keep working unchanged. */
    tm_kbench_dump();

    tp_emit("TPROF|end|ok=1\n");
}

#else  /* !TINYPROF */

void tp_reset(void) { }
void tp_dump(void)  { tp_emit("TPROF|end|ok=0|reason=disabled\n"); }

#endif /* TINYPROF */
