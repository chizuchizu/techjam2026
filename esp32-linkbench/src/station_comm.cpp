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
 * No server/client split: all boards are symmetric peers.
 *
 * Serial commands:
 *   'S'  scan nearby WiFi networks (SSID / RSSI / encryption / channel)
 *   'I'  own IP / MAC / mDNS hostname / known peers
 *   'H'  PING every known peer and print measured RTT (ms)
 *
 * Build-time config (platformio build_flags):
 *   -DLF_WIFI_SSID=\"MYSSID\"  -DLF_WIFI_PASS=\"MYPASS\"   (escaped quotes)
 *   -DLF_FORCE_AP=1   (host LINKNET AP regardless; default 0)
 */
#include <Arduino.h>
#include <cstdio>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>

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

static WiFiUDP udp;
static IPAddress g_peers[LF_MAX_PEERS];
static uint8_t g_npeers = 0;
static uint16_t g_seq[LF_MAX_PEERS] = {0};
static uint32_t g_seq_sent_at[LF_MAX_PEERS] = {0};
static char g_host[32];
static uint32_t g_last_beacon = 0;
static bool g_connected = false;
static bool g_ap_mode = false;

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

static int find_peer(IPAddress ip) {
    for (int i = 0; i < g_npeers; i++) if (g_peers[i] == ip) return i;
    return -1;
}
static int add_peer(IPAddress ip) {
    int i = find_peer(ip);
    if (i >= 0) return i;
    if (g_npeers < LF_MAX_PEERS) { g_peers[g_npeers] = ip; return g_npeers++; }
    return -1;
}

static void print_info() {
    uint8_t mac[6]; WiFi.macAddress(mac);
    Serial.printf("LINKFW-S|info|ip=%s|mac=%02X:%02X:%02X:%02X:%02X:%02X|host=%s.local|"
                  "mode=%s|rssi=%ld|npeers=%d|",
                  our_ip().toString().c_str(),
                  mac[0],mac[1],mac[2],mac[3],mac[4],mac[5],
                  g_host, g_ap_mode ? "AP" : "STA", (long)WiFi.RSSI(), g_npeers);
    for (int i = 0; i < g_npeers; i++) Serial.printf("peer%d=%s|", i, g_peers[i].toString().c_str());
    Serial.println();
}

static void send_beacon() {
    udp.beginPacket(IPAddress(255,255,255,255), LF_UDP_PORT);
    udp.printf("BEACON|%s|%s", g_host, our_ip().toString().c_str());
    udp.endPacket();
}

static void send_ping(IPAddress dst) {
    int i = add_peer(dst);
    if (i < 0) return;
    udp.beginPacket(dst, LF_UDP_PORT);
    uint16_t s = ++g_seq[i];
    g_seq_sent_at[i] = millis();
    g_peers[i] = dst;
    udp.printf("PING|%u", s);
    udp.endPacket();
}

static void handle_rx() {
    int n = udp.parsePacket();
    if (!n) return;
    char buf[96];
    int len = udp.read((uint8_t*)buf, sizeof(buf)-1);
    if (len < 0) len = 0;
    buf[len] = 0;
    IPAddress src = udp.remoteIP();
    int i = add_peer(src);
    g_peers[i] = src;
    Serial.printf("LINKFW-S|rx|from=%s|%s\n", src.toString().c_str(), buf);

    if (strncmp(buf, "PING|", 5) == 0) {
        udp.beginPacket(src, LF_UDP_PORT);
        udp.printf("PONG|%s", buf + 5);
        udp.endPacket();
    } else if (strncmp(buf, "PONG|", 5) == 0) {
        int k = find_peer(src);
        if (k >= 0) {
            uint16_t s = (uint16_t)atoi(buf + 5);
            if (s == g_seq[k]) {
                uint32_t rtt = millis() - g_seq_sent_at[k];
                Serial.printf("LINKFW-S|rtt|from=%s|%u ms\n", src.toString().c_str(), rtt);
            }
        }
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

void setup() {
    Serial.begin(115200);
    delay(100);
    Serial.printf("LINKFW-S|boot|role=STATION|fw=station-comm-v3\n");

    uint8_t mac[6]; WiFi.macAddress(mac);
    snprintf(g_host, sizeof(g_host), "esp32-%02X%02X", mac[4], mac[5]);

    scan_wifi();                 // report what's around (also useful picks SSID)

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

    if (MDNS.begin(g_host)) {
        MDNS.addService("link", "udp", LF_UDP_PORT);
        MDNS.addService("esp32", "udp", LF_UDP_PORT);
        Serial.printf("LINKFW-S|mdns|host=%s.local|port=%d\n", g_host, LF_UDP_PORT);
    } else {
        Serial.println("LINKFW-S|mdns|err");
    }

    udp.begin(LF_UDP_PORT);
    print_info();
    Serial.println("LINKFW-S|READY");
}

void loop() {
    if (g_connected) {
        uint32_t now = millis();
        if (now - g_last_beacon >= LF_BEACON_MS) {
            g_last_beacon = now;
            send_beacon();
        }
    }
    handle_rx();

    if (Serial.available()) {
        char c = (char)Serial.read();
        if (c == 'S' || c == 's') scan_wifi();
        else if (c == 'I' || c == 'i') print_info();
        else if (c == 'H' || c == 'h') {
            Serial.printf("LINKFW-S|ping|targets=%d\n", g_npeers);
            for (int i = 0; i < g_npeers; i++) send_ping(g_peers[i]);
        }
    }
}

#endif // BUILD_ROLE_STATION
