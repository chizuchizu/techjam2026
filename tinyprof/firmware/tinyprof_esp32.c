/*
 * tinyprof_esp32.c - ESP32 platform metrics for tinyprof.
 *
 * Everything here reads state the ESP-IDF / FreeRTOS runtime already keeps,
 * so it adds no static RAM of its own - which is the requirement, since the
 * optimised case-2 build has ~384 B of free DRAM.
 *
 * Compiled only on ESP32; on the host build the weak no-op in tinyprof.c wins,
 * so tools/host_test.c is unaffected.
 */
#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)

#include "tinyprof.h"

#include <stdio.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_cpu.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

void tp_dump_platform(void) {
    char line[192];

    /* Heap. min_free is the watermark since boot, so it survives the fact that
     * a dump happens *after* the forward has already released whatever it used.
     * largest_free is the fragmentation read: free minus largest tells you how
     * much of the remaining heap is unusable as one block. */
    (void)snprintf(line, sizeof line,
        "TPROF|mem|heap_free=%u|heap_min_free=%u|heap_largest=%u\n",
        (unsigned)esp_get_free_heap_size(),
        (unsigned)esp_get_minimum_free_heap_size(),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    tp_emit(line);

    /* Stack high-water mark of the task that ran the forward, in words. This
     * is the only runtime number that says how close the recursion-free but
     * deeply-inlined kernels came to overflowing the 8 KB Arduino loop task. */
    UBaseType_t hwm_words = uxTaskGetStackHighWaterMark(NULL);
    const char* tname = pcTaskGetName(NULL);
    (void)snprintf(line, sizeof line,
        "TPROF|stack|hwm_words=%u|hwm_bytes=%u|task=%s\n",
        (unsigned)hwm_words, (unsigned)(hwm_words * sizeof(StackType_t)),
        tname ? tname : "?");
    tp_emit(line);

    /* Cycle counter sanity: confirms the MHz the analyzer divides by. */
    (void)snprintf(line, sizeof line, "TPROF|clk|ccount=%u\n",
                   (unsigned)esp_cpu_get_cycle_count());
    tp_emit(line);
}

#endif /* ESP32 */
