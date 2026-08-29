/*
 * link.h - direct board-to-board link for the two-node case-2 cluster.
 *
 * No router or credentials are needed: node 0 raises a SoftAP and listens on
 * TCP 5000, node 1 joins it and connects. Both sides then use one persistent
 * TCP socket driven non-blocking from a dedicated FreeRTOS task.
 *
 * The exchange is asynchronous and byte-indexed so it can hide behind compute.
 * A layer's K/V payload is a stream of four per-head chunks; the shard hands
 * chunk h to the link as soon as it is projected (tm_link_tx_ready) and only
 * blocks on the peer's chunk h at the moment attention needs it
 * (tm_link_wait_rx). The link task keeps the socket busy in between, so on a
 * balanced pair the transfer is almost entirely overlapped with arithmetic.
 *
 * Both directions carry the same byte count, so a symmetric swap would
 * deadlock on a blocking write (the TCP window is a few KB against a 32 KB
 * payload); the task interleaves send() and recv() on the raw lwIP socket.
 */
#ifndef TM_LINK_H
#define TM_LINK_H

#include <stdint.h>
#include <stddef.h>

struct TmIoSeg {
    uint8_t* p;
    int      len;
};

bool        tm_link_begin(int role, const char* ssid, const char* pass,
                          uint32_t timeout_ms);
bool        tm_link_up(void);
const char* tm_link_info(void);
const char* tm_link_error(void);

/* Arm an exchange. No payload bytes are sent until tm_link_tx_ready() says
 * how many of them are valid; the receive side starts immediately. */
void tm_link_begin_exchange(const TmIoSeg* tx, int ntx,
                            const TmIoSeg* rx, int nrx);
void tm_link_tx_ready(uint32_t cumulative_bytes);
bool tm_link_wait_rx(uint32_t cumulative_bytes, uint32_t timeout_ms);
bool tm_link_end_exchange(uint32_t timeout_ms);

/* Blocking convenience wrapper (used for the barriers). */
bool tm_link_exchange(const TmIoSeg* tx, int ntx,
                      const TmIoSeg* rx, int nrx, uint32_t timeout_ms);
bool tm_link_barrier(uint8_t tag, uint32_t timeout_ms);
void tm_link_new_forward(void);   /* reset the per-forward exchange epoch */

/* Called roughly every 100 ms while the link blocks, so a board waiting on its
 * peer still writes to USB and the host handle stays alive. */
void tm_link_set_heartbeat(void (*fn)(void));

uint32_t tm_link_bytes_tx(void);
uint32_t tm_link_bytes_rx(void);
void     tm_link_reset_stats(void);
uint32_t tm_link_retx(void);   /* UDP datagrams retransmitted */

/* Raw throughput probe: n symmetric swaps of `bytes` each way, using buf as
 * both source and sink. Returns microseconds, or 0 on failure. */
uint64_t tm_link_bandwidth(uint8_t* tx_buf, uint8_t* rx_buf, int bytes, int n);

#endif /* TM_LINK_H */
