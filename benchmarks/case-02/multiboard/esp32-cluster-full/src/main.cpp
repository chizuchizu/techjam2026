/*
 * main.cpp - ESP32-C3 firmware for the two-board case-2 cluster.
 *
 * One binary serves both nodes; the host assigns the role at runtime, so the
 * pair is flashed identically. Node 0 raises the SoftAP and node 1 joins it
 * (see link.cpp) - no router, no credentials on the benchmark bench.
 *
 * Host serial protocol (native USB CDC, 115200):
 *   'M'                      -> "TM SHARD node=<r> sloc=<n> link=<0|1>"
 *   'N' <'0'|'1'>            -> take that role, bring the peer link up,
 *                               reply "TM LINK <OK|FAIL> <info>"
 *   'L' <32768 B float32 LE> -> load this node's local input rows,
 *                               reply "TM LOADED"
 *   'G'                      -> start barrier, full 4-layer forward, end
 *                               barrier; reply
 *                               "TM DONE us=<wall> comp=<us> link=<us> tx=<B> rx=<B>"
 *   'O'                      -> stream 32768 B of local output rows + "END"
 *   'T' <n>                  -> n back-to-back timed forwards, reply
 *                               "TM TIMES <us> ..."
 *
 * 'G' brackets the forward with peer barriers, so both boards report the same
 * wall window: the reported time is a true end-to-end distributed forward, not
 * a per-node compute time. Host-side scatter/gather of the input and output is
 * outside that window, exactly as the single-board 'R' timing excludes its own
 * serial transfer.
 */
#include <Arduino.h>
#include <esp_task_wdt.h>
#include <esp_timer.h>

#include "kernels.h"
#include "link.h"
#include "model_shard.h"

extern const uint8_t _binary_weights_bin_start[] asm("_binary_weights_bin_start");
extern const uint8_t _binary_weights_q12_bin_start[] asm("_binary_weights_q12_bin_start");

static const char* AP_SSID = "tm-case2-cluster";
static const char* AP_PASS = "case2bench";

static const float* g_wf32 = (const float*)_binary_weights_bin_start;
static TMQ12Weights g_q12;
static TMShard      g_shard;
static int          g_role = -1;

static const size_t LOCAL_BYTES = (size_t)TMS_SLOC * TM_D * 4;

/* Per-layer K/V swap, streamed head by head. The eight segments are
 * [scales_h][K_h V_h] for h = 0..3; head h is complete after
 * (h+1)*TMS_CHUNK_BYTES bytes in either direction. */
static TmIoSeg g_tx_seg[2 * TM_H];
static TmIoSeg g_rx_seg[2 * TM_H];
static uint64_t g_link_us = 0;
static int64_t  g_link_t0 = 0;
static bool     g_link_bad = false;

static void arm_layer_exchange(void) {
    for (int h = 0; h < TM_H; h++) {
        void *s_own, *kv_own, *s_peer, *kv_peer;
        tm_shard_chunk(&g_shard, h, 0, &s_own, &kv_own);
        tm_shard_chunk(&g_shard, h, 1, &s_peer, &kv_peer);
        g_tx_seg[2 * h]     = {(uint8_t*)s_own,  8};
        g_tx_seg[2 * h + 1] = {(uint8_t*)kv_own, TMS_HEAD_ELEMS * 2};
        g_rx_seg[2 * h]     = {(uint8_t*)s_peer,  8};
        g_rx_seg[2 * h + 1] = {(uint8_t*)kv_peer, TMS_HEAD_ELEMS * 2};
    }
    tm_link_begin_exchange(g_tx_seg, 2 * TM_H, g_rx_seg, 2 * TM_H);
}

/* Emitted only while the link is actually blocked (see tm_link_set_heartbeat),
 * as a sign of life for a board that is waiting on its peer. It is deliberately
 * NOT used on the healthy path: a steady dribble of single bytes is what makes
 * the usbip bridge drop the reply that follows. */
static inline void heartbeat(void) {
    /* Never block: if the host handle has stalled the CDC TX buffer fills, and
     * a plain Serial.write() would park the forward until the host came back. */
    if (Serial.availableForWrite() > 16) Serial.write('.');
}

/* hook: head h is projected, release its bytes to the link task */
static void on_kv_sent(void*, int h) {
    tm_link_tx_ready((uint32_t)(h + 1) * TMS_CHUNK_BYTES);
}

/* hook: attention is about to consume head h, so block until it has landed */
static void on_kv_needed(void*, int h) {
    const int64_t t0 = esp_timer_get_time();
    if (!tm_link_wait_rx((uint32_t)(h + 1) * TMS_CHUNK_BYTES, 30000))
        g_link_bad = true;
    g_link_us += (uint64_t)(esp_timer_get_time() - t0);
}

/* Returns the wall time of the distributed forward (barrier to barrier), or 0
 * on link failure. compute_us excludes the time spent blocked on the link. */
static uint32_t cluster_forward(uint32_t* compute_us) {
    g_link_us = 0;
    g_link_bad = false;
    tm_link_new_forward();
    if (!tm_link_barrier(0xA5, 30000)) return 0;
    const int64_t t0 = esp_timer_get_time();

    for (int l = 0; l < TM_L; l++) {
        arm_layer_exchange();
        tm_shard_layer_pre(&g_shard, l);    /* fires on_kv_sent per head   */
        tm_shard_layer_post(&g_shard, l);   /* blocks in on_kv_needed only */
        if (g_link_bad) return 0;
        const int64_t td = esp_timer_get_time();
        if (!tm_link_end_exchange(30000)) return 0;
        g_link_us += (uint64_t)(esp_timer_get_time() - td);
    }
    tm_shard_final(&g_shard);

    const int64_t t1 = esp_timer_get_time();
    if (!tm_link_barrier(0x5A, 30000)) return 0;
    const int64_t t2 = esp_timer_get_time();

    /* g_link_us counts only time actually spent *waiting* on the link; the
     * rest of the transfer overlapped with arithmetic and is invisible here. */
    *compute_us = (uint32_t)((t1 - t0) - (int64_t)g_link_us);
    g_link_us += (uint64_t)(t2 - t1);
    return (uint32_t)(t2 - t0);
}

static void read_local_input(void) {
    uint8_t* p = (uint8_t*)tm_shard_input(&g_shard);
    size_t got = 0;
    while (got < LOCAL_BYTES) {
        int n = Serial.readBytes((char*)(p + got), LOCAL_BYTES - got);
        if (n <= 0) delay(1);
        else got += (size_t)n;
    }
}

void setup() {
    Serial.begin(115200);
    Serial.setRxBufferSize(4096);
    /* a forward runs for seconds without yielding; the interrupt watchdog is
     * untouched, only the idle-task software watchdog is stood down */
    esp_task_wdt_deinit();
    delay(200);
    tm_scan_q12(_binary_weights_q12_bin_start, &g_q12);
    tm_link_set_heartbeat(heartbeat);
    tm_shard_init(&g_shard, 0, g_wf32, &g_q12);
    g_shard.kv_sent = on_kv_sent;
    g_shard.kv_needed = on_kv_needed;
    Serial.printf("TM cluster node ready sloc=%d state=%u B\n",
                  TMS_SLOC, (unsigned)sizeof(TMShard));
}

void loop() {
    while (Serial.available() == 0) delay(2);
    const uint8_t cmd = (uint8_t)Serial.read();

    switch (cmd) {
        case 'M':
            Serial.printf("TM SHARD node=%d sloc=%d link=%d free=%u\n",
                          g_role, TMS_SLOC, tm_link_up() ? 1 : 0,
                          (unsigned)ESP.getFreeHeap());
            break;

        case 'N': {
            while (Serial.available() == 0) delay(2);
            const int role = (Serial.read() == '1') ? 1 : 0;
            g_role = role;
            tm_shard_init(&g_shard, role, g_wf32, &g_q12);
            g_shard.kv_sent = on_kv_sent;
            g_shard.kv_needed = on_kv_needed;
            const bool ok = tm_link_begin(role, AP_SSID, AP_PASS, 40000);
            Serial.printf("TM LINK %s role=%d %s free=%u\n",
                          ok ? "OK" : "FAIL", role, tm_link_info(),
                          (unsigned)ESP.getFreeHeap());
            break;
        }

        case 'L':
            read_local_input();
            Serial.println("TM LOADED");
            break;

        case 'G': {
            uint32_t comp = 0;
            const uint32_t wall = cluster_forward(&comp);
            if (!wall) { Serial.printf("TM ERR link %s\n", tm_link_error()); break; }
            Serial.printf("TM DONE us=%lu comp=%lu link=%lu tx=%lu rx=%lu retx=%lu\n",
                          (unsigned long)wall, (unsigned long)comp,
                          (unsigned long)g_link_us,
                          (unsigned long)tm_link_bytes_tx(),
                          (unsigned long)tm_link_bytes_rx(),
                          (unsigned long)tm_link_retx());
            tm_link_reset_stats();
            break;
        }

        case 'O': {
            /* chunked with an explicit drain: one 32 KB CDC write occasionally
             * loses the tail under a busy radio, and the host can simply ask
             * for the frame again ('O' is idempotent) */
            const uint8_t* p = (const uint8_t*)tm_shard_output(&g_shard);
            for (size_t off = 0; off < LOCAL_BYTES; off += 1024) {
                Serial.write(p + off, 1024);
                Serial.flush();
            }
            Serial.println("END");
            break;
        }

        case 'T': {
            while (Serial.available() == 0) delay(2);
            int n = (int)Serial.read();
            if (n < 1) n = 1;
            if (n > 9) n = 9;
            Serial.print("TM TIMES");
            for (int i = 0; i < n; i++) {
                uint32_t comp = 0;
                const uint32_t wall = cluster_forward(&comp);
                if (!wall) { Serial.print(" ERR"); break; }
                Serial.printf(" %lu/%lu", (unsigned long)wall,
                              (unsigned long)comp);
            }
            Serial.println();
            break;
        }

        case 'B': {
            /* raw link probe: 8 symmetric swaps of one layer's payload */
            const uint64_t us = tm_link_bandwidth((uint8_t*)g_shard.kv_own,
                                                  (uint8_t*)g_shard.kv_peer,
                                                  TMS_KV_BYTES, 8);
            if (!us) { Serial.printf("TM ERR link %s\n", tm_link_error()); break; }
            Serial.printf("TM BW bytes=%d n=8 us=%lu kBps=%lu\n", TMS_KV_BYTES,
                          (unsigned long)us,
                          (unsigned long)((uint64_t)TMS_KV_BYTES * 8 * 1000 / us));
            break;
        }

        case '\n':
        case '\r':
            /* silent keepalive: the host pokes the OUT endpoint while it waits
             * on a reply, which stops the usbip bridge going quiet in one
             * direction for the seconds a distributed forward takes */
            break;

        default:
            Serial.println("TM unknown cmd");
            break;
    }
}
