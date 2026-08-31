#ifdef BUILD_ROLE_STATION
/*
 * esp32-linkfast - STATION mode (same-WiFi IP communication)
 *
 * Flashes IDENTICAL firmware onto ANY number of boards. Every board:
 *   - joins the configured STA WiFi network (same SSID for all boards)
 *   - auto-pins to the strongest matching AP (bssid+channel) so joining is
 *     reliable even against softAPs / ad-hoc networks
 *   - advertises itself via mDNS (esp32-<machlo>.local) so its IP is "known"
 *   - runs a UDP broadcast beacon + listener on UDP port LF_UDP_PORT so every
 *     board auto-discovers every other board's IP on the same network
 *   - answers PING/PONG so you can measure inter-board round-trip over IP
 *   - if the configured network is unavailable, self-hosts an AP named
 *     LF_AP_SSID ("LINKNET") so boards can always talk to each other
 *
 * FLEET layer (station-comm-v4): every board is one node in a fleet.
 *   - stable node id per board: esp32-<mac.low2> (unique per MAC)
 *   - heartbeat BEACON carries id + MAC + uptime + free heap; peers silent for
 *     LF_PEER_TTL_MS are dropped from the registry (node expiry)
 *   - deterministic coordinator election: lowest MAC on the fleet is the
 *     coordinator, everyone else is a worker (no config, same image on every
 *     board); the role is recomputed whenever the fleet changes
 *   - generic JOB/RESULT data path (UDP): the coordinator unicasts a job to
 *     every worker and collects the results - the wire used to split
 *     computation across the fleet. Demo op SUM16 splits a vector and merges
 *     partial sums (press 'B' on the coordinator).
 *   UDP datagrams are capped at 1460 B by Arduino core 3.x NetworkUDP; for
 *   payloads > 1.4 KB use the TCP transport (benchmarks/case-02/multiboard/
 *   esp32_cluster_transport, TCP port 4211).
 *
 * No server/client split: all boards are symmetric peers; the role is a
 * runtime property, not a build flag.
 *
 * Serial commands:
 *   'S'  scan nearby WiFi networks (SSID / RSSI / encryption / channel)
 *   'I'  own IP / MAC / mDNS hostname / role / known peers
 *   'H'  PING every known peer and print measured RTT (ms)
 *   'F'  print the fleet table (id / MAC / IP / role / age / last RTT)
 *   'B'  coordinator: broadcast a SUM16 JOB to all workers and print the
 *        merged result + per-worker RTT (parallel data-path smoke test)
 *
 * Build-time config (platformio build_flags):
 *   -DLF_WIFI_SSID=\"MYSSID\"  -DLF_WIFI_PASS=\"MYPASS\"   (escaped quotes)
 *   -DLF_FORCE_AP=1   (host LINKNET AP regardless; default 0)
 */
#include <Arduino.h>
#include <cstdio>
#include <cstring>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>
#include "esp_mac.h"      // esp_read_mac: read the eFuse base MAC before WiFi init
#if LF_ESPNOW_BENCH
#include <esp_now.h>      // ESP-NOW fast path (Phase-2 8-board transport)
#endif
#ifdef LF_AP_NAT
#include "esp_netif.h"    // core 3.x / IDF 5.1+ only: esp_netif_napt_enable (lwIP NAPT)
#endif

#ifndef LF_WIFI_SSID
#define LF_WIFI_SSID "YOUR_WIFI_SSID"
#endif
#ifndef LF_WIFI_PASS
#define LF_WIFI_PASS "YOUR_WIFI_PASS"
#endif
#ifndef LF_UDP_PORT
#define LF_UDP_PORT 42100
#endif
#ifndef LF_MAX_PEERS
#define LF_MAX_PEERS 16
#endif
#ifndef LF_BEACON_MS
#define LF_BEACON_MS 2000
#endif
#ifndef LF_PEER_TTL_MS
#define LF_PEER_TTL_MS 8000          // drop peers silent for this long (4 beacon gaps)
#endif
#ifndef LF_DGRAM_BUF
#define LF_DGRAM_BUF 1460           // core 3.x NetworkUDP caps a datagram at 1460 B
#endif
#ifndef LF_JOB_TIMEOUT_MS
#define LF_JOB_TIMEOUT_MS 3000
#endif
#ifndef LF_DEMO_VEC_LEN
#define LF_DEMO_VEC_LEN 650         // u16 count; one JOB datagram per worker <= 1300 B
#endif
#ifndef LF_DEMO_AUTO
#define LF_DEMO_AUTO 1              // coordinator auto-runs the compute demo (no PC needed)
#endif
#ifndef LF_DEBUG_RX_PRINT
#define LF_DEBUG_RX_PRINT 0         // 1 = log every UDP rx datagram (serial throttle on the data path)
#endif
#ifdef LF_DEMO_AUTO
#define LF_DEMO_AUTO_PERIOD_MS 6000
#define LF_DEMO_AUTO_STARTUP_MS 8000   // first auto-run shortly after role is known
#endif
#ifndef LF_STA_TIMEOUT_S
#define LF_STA_TIMEOUT_S 15
#endif
#ifndef LF_AP_SSID
#define LF_AP_SSID "LINKNET"
#endif
#ifndef LF_AP_PASS
#define LF_AP_PASS "linkfast123"
#endif
#ifndef LF_FORCE_AP
#define LF_FORCE_AP 0
#endif

// ---- ESP-NOW bench toggles (Phase-2 fast-path transport measurement) ----
#ifndef LF_ESPNOW_BENCH
#define LF_ESPNOW_BENCH 0              // 1 = run the ESP-NOW broadcast dispatch RTT bench
#endif
#ifndef LF_ESPNOW_BENCH_ROUNDS
#define LF_ESPNOW_BENCH_ROUNDS 64      // measurement rounds (after 3 warmup)
#endif
#ifndef LF_ESPNOW_BENCH_INTERVAL_MS
#define LF_ESPNOW_BENCH_INTERVAL_MS 15 // min gap between rounds (reply << 15 ms)
#endif
#ifndef LF_ESPNOW_SLOT_US
#define LF_ESPNOW_SLOT_US 1500         // per-worker collect slot (avoids 8->1 CSMA burst)
#endif

static WiFiUDP udp;

// ---------------- fleet node registry ----------------
typedef struct {
    IPAddress ip;
    uint32_t last_heard;     // millis() of last packet from this peer
    uint32_t seq_sent_at;    // millis() when the last PING seq was sent
    uint16_t seq;            // last PING seq we sent
    uint32_t last_rtt;       // ms, last measured round trip
    char id[24];             // peer node id "esp32-XXXX"
    uint8_t mac[6];          // peer MAC (0 if unknown yet)
} FleetRec;

static FleetRec g_fleet[LF_MAX_PEERS];
static int g_nfleet = 0;
static uint8_t g_mymac[6];
static bool g_role_coord = false;
static bool g_role_known = false;   // false while a peer MAC is still unknown
static int g_last_announced_n = -1;
static char g_host[32];
static uint32_t g_last_beacon = 0;
static bool g_connected = false;
static bool g_ap_mode = false;

// ---------------- SUM16 demo job state ----------------
typedef struct {
    bool active;
    uint16_t jobid;
    uint32_t start_ms;
    uint32_t deadline;
    uint32_t got;
    uint32_t pending;             // workers that have not answered yet
    uint64_t sum;
    uint64_t expected;
    uint32_t sent_at[LF_MAX_PEERS];
    uint32_t rtt[LF_MAX_PEERS];
    uint16_t cnt[LF_MAX_PEERS];
} JobState;
static JobState g_job;
static uint16_t g_job_seq = 0;
static uint16_t g_demo_vec[LF_DEMO_VEC_LEN];
#ifdef LF_AP_NAT
static bool g_napt_enabled = false;
static uint32_t g_sta_join_at = 0;
#endif

static IPAddress our_ip() {
    return g_ap_mode ? WiFi.softAPIP() : WiFi.localIP();
}

static bool ssid_visible() {
    int8_t n = WiFi.scanNetworks();
    bool ok = false;
    for (int8_t i = 0; i < n; i++) {
        if (strcmp(WiFi.SSID(i).c_str(), LF_WIFI_SSID) == 0) { ok = true; break; }
    }
    WiFi.scanDelete();
    return ok;
}

static int mac_cmp(const uint8_t* a, const uint8_t* b) {
    for (int i = 0; i < 6; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// pointer to the idx-th '|'-separated field of s (never scans past payloads)
static char* field_n(char* s, int idx) {
    char* p = s;
    for (int k = 0; k < idx; k++) {
        p = strchr(p, '|');
        if (!p) return NULL;
        p++;
    }
    return p;
}

// copy exactly one '|'-delimited field (no NUL-termination is done by field_n)
static void copy_field(char* dst, size_t n, const char* src) {
    if (n == 0) return;
    size_t len = strcspn(src, "|");
    if (len >= n) len = n - 1;
    memcpy(dst, src, len);
    dst[len] = 0;
}

static int find_fleet(IPAddress ip) {
    for (int i = 0; i < g_nfleet; i++) if (g_fleet[i].ip == ip) return i;
    return -1;
}
static int add_fleet(IPAddress ip) {
    int i = find_fleet(ip);
    if (i >= 0) { g_fleet[i].last_heard = millis(); return i; }
    if (g_nfleet < LF_MAX_PEERS) {
        FleetRec* r = &g_fleet[g_nfleet++];
        r->ip = ip;
        r->last_heard = millis();
        r->seq = 0;
        r->seq_sent_at = 0;
        r->last_rtt = 0;
        r->id[0] = 0;
        memset(r->mac, 0, 6);
        return g_nfleet - 1;
    }
    return -1;
}

static void recompute_role() {
    // I am the coordinator when my MAC <= every peer MAC we know.
    bool coord = true;
    bool all_macs = true;
    for (int i = 0; i < g_nfleet; i++) {
        bool known = false;
        for (int k = 0; k < 6; k++) if (g_fleet[i].mac[k]) known = true;
        if (!known) { all_macs = false; continue; }
        if (mac_cmp(g_mymac, g_fleet[i].mac) > 0) coord = false;
    }
    if (g_nfleet == 0) all_macs = true;   // single-node fleet: I am the coordinator
    bool prev = g_role_coord;
    g_role_coord = coord;
    g_role_known = all_macs;
    if (g_role_coord != prev) {
        Serial.printf("LINKFW-S|role|coord=%d|known=%d|peers=%d\n",
                      g_role_coord ? 1 : 0, g_role_known ? 1 : 0, g_nfleet);
    }
    // announce fleet size changes once per size (helps scan by scripts/humans)
    if (g_nfleet != g_last_announced_n) {
        g_last_announced_n = g_nfleet;
        Serial.printf("LINKFW-S|fleet|n=%d|coord=%d|known=%d|name=%s\n",
                      g_nfleet, g_role_coord ? 1 : 0, g_role_known ? 1 : 0, g_host);
    }
}

static void print_info() {
    Serial.printf("LINKFW-S|info|ip=%s|mac=%02X:%02X:%02X:%02X:%02X:%02X|host=%s.local|"
                  "mode=%s|rssi=%ld|role=%s|known=%d|npeers=%d|",
                  our_ip().toString().c_str(),
                  g_mymac[0],g_mymac[1],g_mymac[2],g_mymac[3],g_mymac[4],g_mymac[5],
                  g_host, g_ap_mode ? "AP" : "STA", (long)WiFi.RSSI(),
                  g_role_coord ? "COORD" : "WORKER", g_role_known ? 1 : 0, g_nfleet);
    for (int i = 0; i < g_nfleet; i++) Serial.printf("peer%d=%s|", i, g_fleet[i].ip.toString().c_str());
    Serial.println();
}

static void print_fleet() {
    Serial.printf("LINKFW-S|fleet|n=%d|role=%s|mine=%s|ip=%s\n",
                  g_nfleet, g_role_coord ? "COORD" : "WORKER",
                  g_host, our_ip().toString().c_str());
    for (int i = 0; i < g_nfleet; i++) {
        uint32_t now = millis();
        uint32_t age = (now >= g_fleet[i].last_heard) ? (now - g_fleet[i].last_heard) : 0;
        const char* p_role = (g_fleet[i].mac[0] && mac_cmp(g_mymac, g_fleet[i].mac) > 0)
                                 ? "COORD" : "WORKER";
        Serial.printf("LINKFW-S|peer|idx=%d|id=%s|mac=%02X:%02X:%02X:%02X:%02X:%02X|ip=%s|age=%u ms|rtt=%u ms|role=%s\n",
                      i, g_fleet[i].id[0] ? g_fleet[i].id : "?",
                      g_fleet[i].mac[0],g_fleet[i].mac[1],g_fleet[i].mac[2],
                      g_fleet[i].mac[3],g_fleet[i].mac[4],g_fleet[i].mac[5],
                      g_fleet[i].ip.toString().c_str(), age, g_fleet[i].last_rtt, p_role);
    }
    if (g_job.active) {
        Serial.printf("LINKFW-S|job|in_flight|id=%u|pending=%u|got=%u|sum=%llu\n",
                      g_job.jobid, g_job.pending, g_job.got, (unsigned long long)g_job.sum);
    }
}

static void send_beacon() {
    udp.beginPacket(IPAddress(255,255,255,255), LF_UDP_PORT);
    udp.printf("BEACON|%s|%s|%02X:%02X:%02X:%02X:%02X:%02X|%lu|%lu",
               g_host, our_ip().toString().c_str(),
               g_mymac[0],g_mymac[1],g_mymac[2],g_mymac[3],g_mymac[4],g_mymac[5],
               (unsigned long)millis(), (unsigned long)ESP.getFreeHeap());
    udp.endPacket();
}

static void send_ping(IPAddress dst) {
    int i = add_fleet(dst);
    if (i < 0) return;
    udp.beginPacket(dst, LF_UDP_PORT);
    uint16_t s = ++g_fleet[i].seq;
    g_fleet[i].seq_sent_at = millis();
    g_fleet[i].last_heard = millis();
    udp.printf("PING|%u", s);
    udp.endPacket();
}

// worker side: handle a SUM16 job and reply with the partial sum
static void handle_job_sum(char* buf, int rx_len) {
    // JOB|<coord>|<jobid>|SUM16|<len>|payload(uint16[], host byte order)
    char* f_coord = field_n(buf, 1);
    char* f_jobid = field_n(buf, 2);
    char* f_op    = field_n(buf, 3);
    char* f_len   = field_n(buf, 4);
    uint8_t* payload = (uint8_t*)field_n(buf, 5);
    if (!f_coord || !f_jobid || !f_op || !f_len || !payload) return;
    if (strncmp(f_op, "SUM16", 5) != 0) return;
    int len = atoi(f_len);
    if (len < 0 || len > 650) return;                  // keep datagram <= ~1.3 KB
    int consumed = (int)((char*)payload - buf) + len * 2;
    if (consumed > rx_len) return;                     // truncated / corrupt
    uint64_t s = 0;
    uint16_t* v = (uint16_t*)payload;
    for (int k = 0; k < len; k++) s += v[k];
    int jobid = atoi(f_jobid);
    udp.beginPacket(udp.remoteIP(), LF_UDP_PORT);
    udp.printf("RESULT|%s|%d|SUM16|%llu|%d", g_host, jobid,
               (unsigned long long)s, len);
    udp.endPacket();
    Serial.printf("LINKFW-S|job|work|jobid=%d|len=%d|sum=%llu|to=%s\n",
                  jobid, len, (unsigned long long)s, f_coord);
}

// worker side: handle a BROADCAST SUM16 job (single frame to all workers).
//   JB|<jobid>|<workers>|<len>|payload(uint16[], host byte order)
// The coordinator sends one datagram for the whole vector; each worker sums
// only its own slice. Slice boundaries are a pure function of (worker index,
// worker count, total). Worker index = (#known peers with MAC < mine) - 1,
// i.e. rank 0 is the coordinator, workers are ranks 1..N-1 - the same MAC
// order the coordinator uses to split. This makes dispatch O(1) in N: one
// frame reaches every worker instead of N unicast datagrams.
static void handle_job_bcast(char* buf, int rx_len) {
    char* f_jobid   = field_n(buf, 1);
    char* f_workers = field_n(buf, 2);
    char* f_len     = field_n(buf, 3);
    uint8_t* payload = (uint8_t*)field_n(buf, 4);
    if (!f_jobid || !f_workers || !f_len || !payload) return;
    int total = atoi(f_len);
    int W = atoi(f_workers);
    if (total < 0 || total > 650 || W < 1) return;
    int consumed = (int)((char*)payload - buf) + total * 2;
    if (consumed > rx_len) return;                  // truncated / corrupt
    int rank = 0;                                   // peers with MAC < mine
    for (int i = 0; i < g_nfleet; i++) {
        bool known = false;
        for (int k = 0; k < 6; k++) if (g_fleet[i].mac[k]) known = true;
        if (known && mac_cmp(g_mymac, g_fleet[i].mac) > 0) rank++;
    }
    int w = rank - 1;                               // 0-based worker index
    if (w < 0 || w >= W) {                          // coordinator or unknown rank
        Serial.printf("LINKFW-S|job|bcast_skip|rank=%d|W=%d\n", rank, W);
        return;
    }
    int base  = total / W;
    int rem   = total % W;
    int start = w * base + (w < rem ? w : rem);     // slice start
    int len   = base + (w < rem ? 1 : 0);           // slice length
    if (start < 0 || start + len > total) return;
    uint64_t s = 0;
    uint16_t* v = (uint16_t*)payload;
    for (int k = start; k < start + len; k++) s += v[k];
    int jobid = atoi(f_jobid);
    udp.beginPacket(udp.remoteIP(), LF_UDP_PORT);
    udp.printf("RESULT|%s|%d|SUM16|%llu|%d", g_host, jobid,
               (unsigned long long)s, len);
    udp.endPacket();
    Serial.printf("LINKFW-S|job|work_bcast|jobid=%d|w=%d|start=%d|len=%d|sum=%llu|to=%s\n",
                  jobid, w, start, len, (unsigned long long)s, udp.remoteIP().toString().c_str());
}

// coordinator side: collect one worker result
static void handle_result_sum(char* buf) {
    // RESULT|<worker>|<jobid>|SUM16|<sum>|<count>
    char* f_wid = field_n(buf, 1);
    char* f_job = field_n(buf, 2);
    char* f_op  = field_n(buf, 3);
    char* f_sum = field_n(buf, 4);
    char* f_cnt = field_n(buf, 5);
    if (!f_wid || !f_job || !f_op || !f_sum || !f_cnt) return;
    if (strncmp(f_op, "SUM16", 5) != 0) return;
    if (!g_job.active) return;
    int jobid = atoi(f_job);
    if (jobid != g_job.jobid) return;
    int idx = find_fleet(udp.remoteIP());
    if (idx < 0) return;
    uint64_t part = strtoull(f_sum, NULL, 10);
    g_job.sum += part;
    g_job.got++;
    g_job.pending--;
    g_job.rtt[idx] = millis() - g_job.sent_at[idx];
    Serial.printf("LINKFW-S|job|res|from=%s|id=%s|jobid=%d|part=%llu|cnt=%s|rtt=%u ms|pending=%u\n",
                  udp.remoteIP().toString().c_str(), f_wid, jobid,
                  (unsigned long long)part, f_cnt, g_job.rtt[idx], g_job.pending);
    if (g_job.pending == 0) {
        bool ok = (g_job.sum == g_job.expected) && (g_job.got > 0);
        Serial.printf("LINKFW-S|job|done|id=%u|workers=%u|sum=%llu|expected=%llu|match=%d|ms=%lu\n",
                      g_job.jobid, g_job.got,
                      (unsigned long long)g_job.sum, (unsigned long long)g_job.expected,
                      ok ? 1 : 0, (unsigned long)(millis() - g_job.start_ms));
        for (int i = 0; i < g_nfleet; i++) {
            Serial.printf("LINKFW-S|job|per_worker|idx=%d|id=%s|rtt=%u ms|cnt=%u\n",
                          i, g_fleet[i].id[0] ? g_fleet[i].id : "?",
                          g_job.rtt[i], g_job.cnt[i]);
        }
        g_job.active = false;
    }
}

static void handle_rx() {
    int n = udp.parsePacket();
    if (!n) return;
    static uint8_t g_rx[LF_DGRAM_BUF];
    int len = udp.read(g_rx, sizeof(g_rx) - 1);
    if (len < 0) len = 0;
    g_rx[len] = 0;
    IPAddress src = udp.remoteIP();
    int i = add_fleet(src);
    if (i < 0) { udp.clear(); return; }      // registry full: drop the datagram
#ifdef LF_DEBUG_RX_PRINT
    if (LF_DEBUG_RX_PRINT) Serial.printf("LINKFW-S|rx|from=%s|%s\n", src.toString().c_str(), (char*)g_rx);
#endif

    if (strncmp((char*)g_rx, "PING|", 5) == 0) {
        udp.beginPacket(src, LF_UDP_PORT);
        udp.printf("PONG|%s", (char*)g_rx + 5);
        udp.endPacket();
    } else if (strncmp((char*)g_rx, "PONG|", 5) == 0) {
        int k = find_fleet(src);
        if (k >= 0) {
            uint16_t s = (uint16_t)atoi((char*)g_rx + 5);
            if (s == g_fleet[k].seq) {
                g_fleet[k].last_rtt = millis() - g_fleet[k].seq_sent_at;
                Serial.printf("LINKFW-S|rtt|from=%s|%u ms\n", src.toString().c_str(), g_fleet[k].last_rtt);
            }
        }
    } else if (strncmp((char*)g_rx, "BEACON|", 7) == 0) {
        // BEACON|<id>|<ip>|<mac>|<uptime>|<heap> -> learn id + MAC
        char* f_id  = field_n((char*)g_rx, 1);
        char* f_mac = field_n((char*)g_rx, 3);
        if (i >= 0 && f_id) {
            copy_field(g_fleet[i].id, sizeof(g_fleet[i].id), f_id);
            if (f_mac) {
                unsigned v[6];
                if (sscanf(f_mac, "%x:%x:%x:%x:%x:%x", &v[0],&v[1],&v[2],&v[3],&v[4],&v[5]) == 6) {
                    for (int k = 0; k < 6; k++) g_fleet[i].mac[k] = (uint8_t)v[k];
                }
            }
        }
        recompute_role();
    } else if (strncmp((char*)g_rx, "JB|", 3) == 0) {
        handle_job_bcast((char*)g_rx, len);
    } else if (strncmp((char*)g_rx, "JOB|", 4) == 0) {
        handle_job_sum((char*)g_rx, len);
    } else if (strncmp((char*)g_rx, "RESULT|", 7) == 0) {
        handle_result_sum((char*)g_rx);
    }
}

static void scan_wifi() {
    Serial.println("LINKFW-S|scan|start");
    int8_t n = WiFi.scanNetworks();
    for (int8_t i = 0; i < n; i++) {
        Serial.printf("LINKFW-S|scan|%d|%s|%ld dBm|enc=%d|ch=%d|bssid=%s\n",
                      (int)i, WiFi.SSID(i).c_str(), (long)WiFi.RSSI(i),
                      (int)WiFi.encryptionType(i), (int)WiFi.channel(i),
                      WiFi.BSSIDstr(i).c_str());
    }
    Serial.printf("LINKFW-S|scan|done|found=%d\n", (int)n);
    WiFi.scanDelete();
}

static void join_sta() {
    // auto-pin to the strongest visible AP of our SSID for robust association
    uint8_t b[6] = {0,0,0,0,0,0};
    int ch = 0;
    int8_t n = WiFi.scanNetworks();
    bool pinned = false;
    for (int8_t i = 0; i < n; i++) {
        if (strcmp(WiFi.SSID(i).c_str(), LF_WIFI_SSID) == 0) {
            unsigned v[6];
            sscanf(WiFi.BSSIDstr(i).c_str(), "%x:%x:%x:%x:%x:%x",
                   &v[0],&v[1],&v[2],&v[3],&v[4],&v[5]);
            for (int k = 0; k < 6; k++) b[k] = (uint8_t)v[k];
            ch = WiFi.channel(i);
            pinned = true;
            Serial.printf("LINKFW-S|conn_pin|ssid=%s|bssid=%s|ch=%d\n",
                          LF_WIFI_SSID, WiFi.BSSIDstr(i).c_str(), ch);
            break;
        }
    }
    WiFi.scanDelete();

    WiFi.disconnect();          // clear stale post-scan state
    delay(200);
    if (pinned) WiFi.begin(LF_WIFI_SSID, LF_WIFI_PASS, ch, b);
    else        WiFi.begin(LF_WIFI_SSID, LF_WIFI_PASS);
    Serial.printf("LINKFW-S|conn|ssid=%s%s\n", LF_WIFI_SSID, pinned ? " (pinned)" : "");
}

static void start_ap() {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(LF_AP_SSID, LF_AP_PASS);
    g_ap_mode = true;
    g_connected = true;
    Serial.printf("LINKFW-S|ap|ssid=%s|ip=%s\n", LF_AP_SSID, WiFi.softAPIP().toString().c_str());
}

#ifdef LF_AP_NAT
// --- NAT router support (LF_AP_NAT / core 3.x + IDF 5.1): the board hosts
//     LINKNET for its clients AND joins a configured uplink WiFi (the WAN);
//     NAPT masquerades LINKNET clients out through the uplink so they reach
//     the internet. ------------------
static void nat_join_uplink() {
    uint8_t b[6] = {0,0,0,0,0,0};
    int ch = 0;
    WiFi.disconnect();                 // clear stale post-scan state
    delay(200);
    int8_t n = WiFi.scanNetworks();
    bool pinned = false;
    for (int8_t i = 0; i < n; i++) {
        if (strcmp(WiFi.SSID(i).c_str(), LF_WIFI_SSID) == 0) {
            unsigned v[6];
            sscanf(WiFi.BSSIDstr(i).c_str(), "%x:%x:%x:%x:%x:%x",
                   &v[0],&v[1],&v[2],&v[3],&v[4],&v[5]);
            for (int k = 0; k < 6; k++) b[k] = (uint8_t)v[k];
            ch = WiFi.channel(i);
            pinned = true;
            Serial.printf("LINKFW-S|nat|uplink_pin|ssid=%s|bssid=%s|ch=%d\n",
                          LF_WIFI_SSID, WiFi.BSSIDstr(i).c_str(), ch);
            break;
        }
    }
    WiFi.scanDelete();
    if (pinned) WiFi.begin(LF_WIFI_SSID, LF_WIFI_PASS, ch, b);
    else        WiFi.begin(LF_WIFI_SSID, LF_WIFI_PASS);
    Serial.printf("LINKFW-S|nat|uplink_begin|ssid=%s\n", LF_WIFI_SSID);
}

static void nat_poll() {
    if (WiFi.status() == WL_CONNECTED) {
        if (!g_napt_enabled) {
            esp_netif_t *ap = esp_netif_get_handle_from_ifkey("WIFI_AP_DEF");
            bool ok = (ap && esp_netif_napt_enable(ap) == ESP_OK);
            esp_netif_dns_info_t dns; memset(&dns, 0, sizeof(dns));
            bool have_dns = false;
            // hand LINKNET clients a working DNS server (upstream's if possible)
            esp_netif_t *sta = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
            if (sta && esp_netif_get_dns_info(sta, ESP_NETIF_DNS_MAIN, &dns) == ESP_OK)
                have_dns = true;
            if (!have_dns) {
                dns.ip.type = ESP_IPADDR_TYPE_V4;
                dns.ip.u_addr.ip4.addr = (uint32_t)IPAddress(8, 8, 8, 8);
            }
            esp_netif_set_dns_info(ap, ESP_NETIF_DNS_MAIN, &dns);
            if (ok) {
                g_napt_enabled = true;
                uint8_t *o = (uint8_t*)&dns.ip.u_addr.ip4.addr;
                Serial.printf("LINKFW-S|nat|enabled|ap=%s|wan=%s|dns=%u.%u.%u.%u\n",
                              WiFi.softAPIP().toString().c_str(),
                              WiFi.localIP().toString().c_str(), o[0],o[1],o[2],o[3]);
            } else {
                Serial.printf("LINKFW-S|nat|enable_fail|st=%d\n", (int)WiFi.status());
            }
        }
    } else if (millis() - g_sta_join_at >= 15000) {
        g_sta_join_at = millis();
        nat_join_uplink();             // keep retrying the WAN in the background
    }
}
#endif

static void drop_stale() {
    uint32_t now = millis();
    for (int i = g_nfleet - 1; i >= 0; i--) {
        if (now - g_fleet[i].last_heard > LF_PEER_TTL_MS) {
            Serial.printf("LINKFW-S|peer_gone|%s|%s\n", g_fleet[i].id,
                          g_fleet[i].ip.toString().c_str());
            g_fleet[i] = g_fleet[g_nfleet - 1];
            g_nfleet--;
            recompute_role();
        }
    }
}

// ---------------- SUM16 fleet demo (coordinator side) ----------------
// Broadcast the whole vector once; every worker sums its own MAC-ranked slice
// (see handle_job_bcast). One datagram replaces N unicast JOBs -> O(1) dispatch.
static void send_job_bcast(uint16_t jobid, uint16_t workers, uint16_t total) {
    udp.beginPacket(IPAddress(255, 255, 255, 255), LF_UDP_PORT);
    udp.printf("JB|%u|%u|%u|", jobid, workers, total);
    udp.write((const uint8_t*)g_demo_vec, (size_t)total * 2);
    udp.endPacket();
}

static void start_demo_sum() {
    if (!g_role_coord || !g_role_known) {
        Serial.printf("LINKFW-S|job|not_coord|coord=%d|known=%d|peers=%d\n",
                      g_role_coord ? 1 : 0, g_role_known ? 1 : 0, g_nfleet);
        Serial.println("LINKFW-S|job|hint|run this on the lowest-MAC board");
        return;
    }
    if (g_nfleet < 1) { Serial.println("LINKFW-S|job|no_peers"); return; }
    if (g_job.active) { Serial.println("LINKFW-S|job|busy"); return; }
    // deterministic pseudo-random vector so the coordinator can verify the merge
    uint32_t x = 0x1234ABCDu;
    uint64_t expected = 0;
    for (int k = 0; k < LF_DEMO_VEC_LEN; k++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        g_demo_vec[k] = (uint16_t)(x >> 8);
        expected += g_demo_vec[k];
    }
    uint16_t jobid = ++g_job_seq;
    int base = LF_DEMO_VEC_LEN / g_nfleet;
    if (base < 1) base = 1;
    // one datagram per worker; keep each <= 650 u16 (~1300 B payload)
    if (base + (LF_DEMO_VEC_LEN % g_nfleet ? 1 : 0) > 650) {
        Serial.printf("LINKFW-S|job|oversize|demand %d u16/worker > 650\n",
                      base + (LF_DEMO_VEC_LEN % g_nfleet ? 1 : 0));
        return;
    }
    g_job.active = true;
    g_job.jobid = jobid;
    g_job.got = 0;
    g_job.pending = g_nfleet;
    g_job.sum = 0;
    g_job.expected = expected;
    g_job.start_ms = millis();
    g_job.deadline = g_job.start_ms + LF_JOB_TIMEOUT_MS;
    for (int i = 0; i < g_nfleet; i++) {
        g_job.cnt[i] = base;
        g_job.rtt[i] = 0;
    }
    send_job_bcast(jobid, (uint16_t)g_nfleet, (uint16_t)LF_DEMO_VEC_LEN);
    uint32_t now0 = millis();
    for (int i = 0; i < g_nfleet; i++) g_job.sent_at[i] = now0;
    Serial.printf("LINKFW-S|job|start|id=%u|workers=%d|len=%u|expected=%llu\n",
                  jobid, g_nfleet, LF_DEMO_VEC_LEN, (unsigned long long)expected);
}

#ifdef LF_DEMO_AUTO
static uint32_t g_next_demo_ms = 0;
static void schedule_demo() {
    if (!g_role_coord || !g_role_known || g_nfleet < 1 || g_job.active) return;
    if (g_next_demo_ms == 0) g_next_demo_ms = millis() + LF_DEMO_AUTO_STARTUP_MS;
    if (millis() >= g_next_demo_ms) {
        g_next_demo_ms = millis() + LF_DEMO_AUTO_PERIOD_MS;
        start_demo_sum();
    }
}
#endif

static void poll_job() {
    if (!g_job.active) return;
    if (millis() < g_job.deadline) return;
    Serial.printf("LINKFW-S|job|timeout|id=%u|pending=%u|got=%u|sum=%llu|expected=%llu\n",
                  g_job.jobid, g_job.pending, g_job.got,
                  (unsigned long long)g_job.sum, (unsigned long long)g_job.expected);
    g_job.active = false;
}


// ---------------- ESP-NOW fast-path bench (LF_ESPNOW_BENCH) ----------------
// Hardware-validates the 8-board comm transport. The coordinator broadcast-
// dispatches one tiny control frame over ESP-NOW (O(1), one frame to ALL
// workers); each worker replies after its slot (worker_index * SLOT_US) to
// avoid the 8->1 CSMA "thundering herd". We measure the true broadcast->reply
// round trip in us on THIS hardware. Baseline: the fleet UDP PING/PONG (ms).
#if LF_ESPNOW_BENCH
static uint8_t  g_eb_bcast[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
static bool     g_eb_on = false;
static bool     g_eb_done = false;         // bench already ran once
static bool     g_eb_ready = false;        // roles known + fleet seen
static uint32_t g_eb_ready_at = 0;
static uint16_t g_eb_seq = 0;
static bool     g_eb_pending = false;
static uint32_t g_eb_sent_us = 0;
static uint32_t g_eb_sent_ms = 0;          // stale-reply watchdog
#ifndef LF_ESPNOW_TIMEOUT_MS
#define LF_ESPNOW_TIMEOUT_MS 200
#endif
static uint32_t g_eb_round = 0;
static uint32_t g_eb_last = 0;
static uint32_t g_eb_rtt_min = 0xFFFFFFFFu;
static uint32_t g_eb_rtt_max = 0;
static uint64_t g_eb_rtt_sum = 0;
static int      g_eb_rtt_n = 0;
static uint32_t g_eb_rtt_vals[LF_ESPNOW_BENCH_ROUNDS + 8];
static int      g_eb_miss = 0;
static const int g_eb_warmup = 3;
static bool     g_eb_defer_on = false;
static uint16_t g_eb_defer_seq = 0;
static uint8_t  g_eb_defer_mac[6];
static uint32_t g_eb_defer_slot_us = 0;
static uint32_t g_eb_defer_us = 0;

static bool mac_eq(const uint8_t *a, const uint8_t *b) {
    for (int i = 0; i < 6; i++) if (a[i] != b[i]) return false;
    return true;
}
static bool mac_is_zero(const uint8_t *a) {
    for (int i = 0; i < 6; i++) if (a[i]) return false;
    return true;
}

static int my_worker_index() {
    // number of fleet members with a live MAC ordering below ours = our rank,
    // minus one for the coordinator itself -> 0-based worker index.
    int rank = 0;
    for (int i = 0; i < g_nfleet; i++) {
        if (mac_is_zero(g_fleet[i].mac)) continue;
        if (mac_cmp(g_mymac, g_fleet[i].mac) > 0) rank++;
    }
    return rank - 1;                        // 0-based, -1 if called on the coordinator
}

static void eb_add_peer(const uint8_t *mac, int step) {
    if (esp_now_is_peer_exist(mac)) return;
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, mac, 6);
    peer.channel = 0;                       // current channel (both boards on AP ch 1)
    peer.ifidx = g_ap_mode ? (wifi_interface_t)WIFI_IF_AP : (wifi_interface_t)WIFI_IF_STA;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
}

static void eb_ensure_bcast_peer() { eb_add_peer(g_eb_bcast, 1); }

static uint8_t *eb_peer_target() {
    // AP board: first fleet station MAC; STA board: the AP's BSSID we joined.
    if (g_ap_mode) {
        if (g_nfleet > 0 && !mac_is_zero(g_fleet[0].mac)) return g_fleet[0].mac;
        return NULL;
    }
    return WiFi.BSSID();   // we joined this AP -> its MAC is the AP peer
}

// parse a decimal uint16 from data[from..]; returns false if none found
static bool eb_parse_uint16(const uint8_t *data, int len, int from, uint16_t *out) {
    uint16_t v = 0; int d = 0;
    for (int i = from; i < len; i++) {
        if (data[i] < '0' || data[i] > '9') break;
        v = v * 10 + (uint16_t)(data[i] - '0');
        d++;
    }
    if (!d) return false;
    *out = v;
    return true;
}

// worker: "EB|<seq>" broadcast from the coordinator -> slotted unicast "EBR|<seq>"
// reply to the sender (slots = worker_index * LF_ESPNOW_SLOT_US, kept tiny so an
// 8-node storm of simultaneous replies never reaches the coordinator at once).
static void eb_worker_on_dispatch(const uint8_t *mac, const uint8_t *data, int len) {
    // IMPORTANT: never send from inside the recv callback (lwIP/esp_now mutex
    // deadlock on this stack). We only record what needs replying; eb_poll
    // (the main loop) performs the slot-timed unicast send.
    if (!g_eb_ready) return;
    uint16_t seq = 0;
    int from = 2;
    if (from < len && data[from] == '|') from++;     // "EB|" prefix
    if (!eb_parse_uint16(data, len, from, &seq)) return;
    int w = my_worker_index(); if (w < 0) w = 0;
    if (g_eb_defer_on) return;                       // reply already queued for this round
#ifdef LF_ESPNOW_DEBUG
    if (seq <= 8) Serial.printf("LINKFW-E|w|dispatch|seq=%u|slot_us=%u\n", seq, (uint32_t)w * (uint32_t)LF_ESPNOW_SLOT_US);
#endif
    g_eb_defer_on = true;
    g_eb_defer_seq = seq;
    memcpy(g_eb_defer_mac, mac, 6);
    g_eb_defer_slot_us = (uint32_t)w * (uint32_t)LF_ESPNOW_SLOT_US;
    g_eb_defer_us = micros() + g_eb_defer_slot_us;
}

static void eb_on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    const uint8_t *mac = info ? info->src_addr : NULL;
#ifdef LF_ESPNOW_DEBUG
#ifndef LF_ESPNOW_DIAG_PING
    {
        static int dbg1 = 0;
        if (dbg1 < 8) {
            dbg1++;
            Serial.printf("LINKFW-E|%s|raw|n=%d|len=%d|t_ms=%u|src=%02X:%02X:%02X:%02X:%02X:%02X|b0=%02X\n",
                          (g_role_coord ? "c" : "w"), dbg1, len, (uint32_t)millis(),
                          mac[0],mac[1],mac[2],mac[3],mac[4],mac[5], data[0]);
        }
    }
#endif
#endif
#ifdef LF_ESPNOW_DIAG_PING
    {
        static int ping_rx = 0;
        if (ping_rx < 40) {
            char hexb[32]; hexb[0] = 0;
            for (int q = 0; q < len && q < 10; q++) {
                snprintf(hexb + strlen(hexb), sizeof(hexb) - strlen(hexb), "%02X", data[q]);
            }
            ping_rx++;
            Serial.printf("LINKFW-E|ping|rx|n=%d|len=%d|t_ms=%u|src=%02X:%02X:%02X:%02X:%02X:%02X|d=%s\n",
                          ping_rx, len, (uint32_t)millis(),
                          mac[0],mac[1],mac[2],mac[3],mac[4],mac[5], hexb);
        }
    }
#endif
    if (len < 2 || data[0] != 'E' || data[1] != 'B') return;
    if (g_role_coord) {                     // coordinator: match an EBR reply to a sent seq
        if (len < 4 || data[2] != 'R') return;
        uint16_t seq = 0;
        int cfrom = 2;
        if (cfrom < len && data[cfrom] == 'R') cfrom++;             // "EBR" prefix
        if (cfrom < len && data[cfrom] == '|') cfrom++;             // "EBR|"
        if (!eb_parse_uint16(data, len, cfrom, &seq)) return;
        if (!g_eb_pending || seq != g_eb_seq) return;
        uint32_t rtt = micros() - g_eb_sent_us;
#ifdef LF_ESPNOW_DEBUG
        if (seq <= 8) {
            Serial.printf("LINKFW-E|b|rx|seq=%u|rtt_us=%lu|t_ms=%u\n",
                          seq, (unsigned long)rtt, (uint32_t)millis());
        }
#endif
        g_eb_pending = false;
        g_eb_rtt_vals[g_eb_round] = rtt;    // seq == round (0-based; warmup is rounds 0..2)
        if (g_eb_round >= (uint32_t)g_eb_warmup && g_eb_round - g_eb_warmup < LF_ESPNOW_BENCH_ROUNDS) {
            if (rtt < g_eb_rtt_min) g_eb_rtt_min = rtt;
            if (rtt > g_eb_rtt_max) g_eb_rtt_max = rtt;
            g_eb_rtt_sum += rtt;
            g_eb_rtt_n++;
        }
        return;
    }
    eb_worker_on_dispatch(mac, data, len);  // worker path
}

static void eb_setup() {
    if (esp_now_init() != ESP_OK) {
        Serial.println("LINKFW-E|init|fail");
        return;
    }
    esp_now_register_recv_cb(eb_on_recv);
    Serial.println("LINKFW-E|init|ok");
}

static void eb_send_round() {
    char out[24];
    int n = snprintf(out, sizeof(out), "EB|%u", (uint32_t)g_eb_seq);
    eb_ensure_bcast_peer();
    g_eb_sent_us = micros();
    g_eb_sent_ms = millis();
    g_eb_pending = true;
    esp_now_send(g_eb_bcast, (const uint8_t*)out, (size_t)n);
#ifdef LF_ESPNOW_DEBUG
    if (g_eb_seq <= 8) {
        Serial.printf("LINKFW-E|b|tx|seq=%u|t_ms=%u\n",
                      (uint32_t)g_eb_seq, (uint32_t)g_eb_sent_ms);
    }
#endif
}

static void eb_poll() {
    if (LF_ESPNOW_BENCH == 0) return;
#ifdef LF_ESPNOW_DIAG_PING
    {   // bidirectional ping: both boards broadcast every 300ms, log every rx
        uint32_t nw = millis();
        if (g_eb_ready_at > 0 && nw >= g_eb_ready_at) {
            if (nw - g_eb_last >= 300) {
                g_eb_last = nw;
                if (g_eb_seq <= 40) {
                    char o[24];
                    int nn = snprintf(o, sizeof(o), "EB|P%u", (uint32_t)g_eb_seq);
                    eb_ensure_bcast_peer();
                    esp_now_send(g_eb_bcast, (const uint8_t*)o, (size_t)nn);
                    uint8_t *u = eb_peer_target();
                    if (u) {
                        char ou[24];
                        int un = snprintf(ou, sizeof(ou), "EB|U%u", (uint32_t)g_eb_seq);
                        int urc = esp_now_is_peer_exist(u) ? 1 : 0;
                        eb_add_peer(u, 3);
                        esp_err_t uer = esp_now_send(u, (const uint8_t*)ou, (size_t)un);
                        Serial.printf("LINKFW-E|ping|utx|n=%u|rc=%d|peer=%d\n",
                                      (uint32_t)g_eb_seq, (int)uer, urc);
                    } else {
                        Serial.println("LINKFW-E|ping|utx|no_peer");
                    }
                    Serial.printf("LINKFW-E|ping|tx|n=%u|t_ms=%u\n",
                                  (uint32_t)g_eb_seq, (uint32_t)nw);
                    g_eb_seq++;
                }
            }
        } else if (g_eb_ready_at == 0 && g_role_known) {
            g_eb_ready_at = millis() + 8000;   // settle then ping for 20 rounds
        }
    }
#endif
    if (g_eb_done) return;
    if (!g_eb_ready) {
        if (g_role_known && g_nfleet >= 1 && !g_eb_on) {
            g_eb_ready = true;
            g_eb_ready_at = millis() + 5000;    // settle period after roles are known
        }
        return;
    }
    uint32_t now = millis();
    if (now < g_eb_ready_at) return;
    if (!g_role_coord) {
        // worker: flush the deferred reply when its slot arrives. Send happens in
        // the main loop on purpose -- esp_now_send inside the recv callback can
        // block/deadlock the wifi task on this stack.
        if (g_eb_defer_on && (int32_t)(micros() - g_eb_defer_us) >= 0) {
            g_eb_defer_on = false;
            char out[24];
            int n = snprintf(out, sizeof(out), "EBR|%u", (uint32_t)g_eb_defer_seq);
            eb_add_peer(g_eb_defer_mac, 2);
            esp_now_send(g_eb_defer_mac, (const uint8_t*)out, (size_t)n);
#ifdef LF_ESPNOW_DEBUG
            if (g_eb_defer_seq <= 8) Serial.printf("LINKFW-E|w|send|seq=%u\n", (uint32_t)g_eb_defer_seq);
#endif
        }
        return;
    }
    if (!g_eb_on) {
        g_eb_on = true;
        g_eb_last = 0;
        Serial.printf("LINKFW-E|bench|start|rounds=%d|slot_us=%u\n",
                      LF_ESPNOW_BENCH_ROUNDS, (uint32_t)LF_ESPNOW_SLOT_US);
        return;
    }
    if (g_eb_pending) {
        if (now - g_eb_sent_ms > (uint32_t)LF_ESPNOW_TIMEOUT_MS) {   // lost reply
            g_eb_pending = false;
            g_eb_miss++;
            g_eb_round++;
            Serial.printf("LINKFW-E|round|seq=%u|MISS\n", (uint32_t)g_eb_seq);
        }
        return;
    }
    if (now - g_eb_last < (uint32_t)LF_ESPNOW_BENCH_INTERVAL_MS) return;
    g_eb_last = now;
    if (g_eb_round >= (uint32_t)LF_ESPNOW_BENCH_ROUNDS + (uint32_t)g_eb_warmup) {
        // summarize: min/avg/max + median over the recorded rounds
        uint32_t med = 0;
        if (g_eb_rtt_n > 0) {
            uint32_t arr[LF_ESPNOW_BENCH_ROUNDS + 8];
            int m = 0;
            for (int k = g_eb_warmup; k < (int)g_eb_round; k++) {
                if (g_eb_rtt_vals[k]) arr[m++] = g_eb_rtt_vals[k];
            }
            for (int a = 1; a < m; a++) {           // insertion sort (m <= 64)
                uint32_t key = arr[a]; int b = a - 1;
                while (b >= 0 && arr[b] > key) { arr[b + 1] = arr[b]; b--; }
                arr[b + 1] = key;
            }
            med = arr[m / 2];
        }
        Serial.printf("LINKFW-E|bench|done|n=%d|min_us=%lu|avg_us=%lu|max_us=%lu|med_us=%lu|miss=%d\n",
                      g_eb_rtt_n,
                      (unsigned long)(g_eb_rtt_n ? g_eb_rtt_min : 0),
                      (unsigned long)(g_eb_rtt_n ? (uint32_t)(g_eb_rtt_sum / g_eb_rtt_n) : 0),
                      (unsigned long)(g_eb_rtt_n ? g_eb_rtt_max : 0),
                      (unsigned long)med,
                      g_eb_miss);
        g_eb_on = false;
        g_eb_done = true;
        return;
    }
    g_eb_seq = (uint16_t)(g_eb_round + 1);
    eb_send_round();
    g_eb_round++;
}
#endif // LF_ESPNOW_BENCH

void setup() {
    Serial.begin(115200);
    delay(100);
#ifdef LF_AP_NAT
    Serial.printf("LINKFW-S|boot|role=STATION-NAT|fw=station-comm-v4-fleet\n");
#else
    Serial.printf("LINKFW-S|boot|role=STATION|fw=station-comm-v4-fleet\n");
#endif

    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);   // eFuse base MAC: stable + unique per board
    memcpy(g_mymac, mac, 6);
    snprintf(g_host, sizeof(g_host), "esp32-%02X%02X", mac[4], mac[5]);

    scan_wifi();                 // report what's around (also useful picks SSID)

#ifdef LF_AP_NAT
    // NAT router: host LINKNET (192.168.4.1) AND join the uplink WiFi (WAN).
    WiFi.mode(WIFI_AP_STA);
    WiFi.setSleep(false);
    WiFi.softAP(LF_AP_SSID, LF_AP_PASS);
    g_ap_mode = true;
    g_connected = true;
    Serial.printf("LINKFW-S|nat|ap ssid=%s|ip=%s\n", LF_AP_SSID, WiFi.softAPIP().toString().c_str());
    g_sta_join_at = millis();
    nat_join_uplink();
#else
    if (LF_FORCE_AP) {
        start_ap();
    } else {
        WiFi.mode(WIFI_STA);
        WiFi.setSleep(false);
        join_sta();
        // wait for the network; retry pinned join if it appears mid-wait
        // (handles a partner board booting up its own AP at the same time)
        int attempts = 0;
        while (attempts < 3) {
            int waited = 0;
            while (WiFi.status() != WL_CONNECTED && waited < 5000) { delay(250); waited += 250; }
            if (WiFi.status() == WL_CONNECTED) break;
            attempts++;
            if (ssid_visible()) join_sta(); else break;
        }
        if (WiFi.status() == WL_CONNECTED) {
            Serial.printf("LINKFW-S|connected|ip=%s|rssi=%ld\n",
                          WiFi.localIP().toString().c_str(), (long)WiFi.RSSI());
            g_connected = true;
        } else {
            Serial.printf("LINKFW-S|sta_failed|status=%d\n", (int)WiFi.status());
            start_ap();          // couldn't find the network -> host our own
        }
    }
#endif // LF_AP_NAT

    if (MDNS.begin(g_host)) {
        MDNS.addService("link", "udp", LF_UDP_PORT);
        MDNS.addService("esp32", "udp", LF_UDP_PORT);
        Serial.printf("LINKFW-S|mdns|host=%s.local|port=%d\n", g_host, LF_UDP_PORT);
    } else {
        Serial.println("LINKFW-S|mdns|err");
    }

    udp.begin(LF_UDP_PORT);
#if LF_ESPNOW_BENCH
    eb_setup();
#endif
    print_info();
    Serial.println("LINKFW-S|READY");
}

void loop() {
#ifdef LF_AP_NAT
    nat_poll();
#endif
    if (g_connected) {
        uint32_t now = millis();
        if (now - g_last_beacon >= LF_BEACON_MS) {
            g_last_beacon = now;
            send_beacon();
        }
    }
    handle_rx();
    drop_stale();          // registry expiry: remove silent peers
    poll_job();            // coordinator: fail a demo that overran its deadline
#ifdef LF_DEMO_AUTO
    schedule_demo();       // coordinator: auto-run the parallel compute demo
#endif

    if (Serial.available()) {
        char c = (char)Serial.read();
        if (c == 'S' || c == 's') scan_wifi();
        else if (c == 'I' || c == 'i') print_info();
        else if (c == 'F' || c == 'f') print_fleet();
        else if (c == 'B' || c == 'b') start_demo_sum();
        else if (c == 'H' || c == 'h') {
            Serial.printf("LINKFW-S|ping|targets=%d\n", g_nfleet);
            for (int i = 0; i < g_nfleet; i++) send_ping(g_fleet[i].ip);
        }
    }
#if LF_ESPNOW_BENCH
    eb_poll();
#endif
}

#endif // BUILD_ROLE_STATION
