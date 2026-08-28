#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include "secrets.h"

namespace {

constexpr uint32_t MAGIC = 0x45535033u;  // "ESP3"
constexpr uint8_t PROTOCOL_VERSION = 1;
constexpr uint8_t TYPE_ECHO_REQUEST = 1;
constexpr uint8_t TYPE_ECHO_REPLY = 2;
constexpr uint8_t TYPE_DISCOVER_REQUEST = 3;
constexpr uint8_t TYPE_DISCOVER_REPLY = 4;
constexpr uint16_t UDP_PORT = 4210;
constexpr uint16_t TCP_PORT = 4211;
constexpr size_t HEADER_BYTES = 12;
constexpr size_t MAX_UDP_PAYLOAD = 1400;
constexpr size_t MAX_TCP_PAYLOAD = 32768;
constexpr unsigned long WIFI_RETRY_INTERVAL_MS = 10000;
constexpr unsigned long TCP_READ_TIMEOUT_MS = 3000;

WiFiUDP udp;
WiFiServer tcp_server(TCP_PORT);
WiFiClient tcp_client;
uint8_t udp_buffer[HEADER_BYTES + MAX_UDP_PAYLOAD];
uint8_t *tcp_payload = nullptr;
unsigned long last_wifi_retry = 0;

uint16_t readU16(const uint8_t *source) {
  return static_cast<uint16_t>(source[0]) << 8 |
         static_cast<uint16_t>(source[1]);
}

uint32_t readU32(const uint8_t *source) {
  return static_cast<uint32_t>(source[0]) << 24 |
         static_cast<uint32_t>(source[1]) << 16 |
         static_cast<uint32_t>(source[2]) << 8 |
         static_cast<uint32_t>(source[3]);
}

void writeU16(uint8_t *destination, uint16_t value) {
  destination[0] = static_cast<uint8_t>(value >> 8);
  destination[1] = static_cast<uint8_t>(value);
}

void writeU32(uint8_t *destination, uint32_t value) {
  destination[0] = static_cast<uint8_t>(value >> 24);
  destination[1] = static_cast<uint8_t>(value >> 16);
  destination[2] = static_cast<uint8_t>(value >> 8);
  destination[3] = static_cast<uint8_t>(value);
}

void writeHeader(uint8_t *destination,
                 uint8_t type,
                 uint16_t payload_bytes,
                 uint32_t request_id) {
  writeU32(destination, MAGIC);
  destination[4] = PROTOCOL_VERSION;
  destination[5] = type;
  writeU16(destination + 6, payload_bytes);
  writeU32(destination + 8, request_id);
}

bool validHeader(const uint8_t *header,
                 uint8_t expected_type,
                 size_t maximum_payload,
                 uint16_t &payload_bytes,
                 uint32_t &request_id) {
  payload_bytes = readU16(header + 6);
  request_id = readU32(header + 8);
  return readU32(header) == MAGIC && header[4] == PROTOCOL_VERSION &&
         header[5] == expected_type && payload_bytes <= maximum_payload;
}

bool readExact(WiFiClient &client,
               uint8_t *destination,
               size_t bytes,
               unsigned long timeout_ms) {
  size_t received = 0;
  const unsigned long deadline = millis() + timeout_ms;
  while (received < bytes &&
         static_cast<long>(deadline - millis()) > 0 && client.connected()) {
    const int available = client.available();
    if (available > 0) {
      const size_t wanted = bytes - received;
      const int count = client.read(destination + received, wanted);
      if (count > 0) {
        received += static_cast<size_t>(count);
      }
    } else {
      delay(1);
    }
  }
  return received == bytes;
}

bool writeExact(WiFiClient &client, const uint8_t *source, size_t bytes) {
  size_t written = 0;
  while (written < bytes && client.connected()) {
    const size_t count = client.write(source + written, bytes - written);
    if (count == 0) {
      delay(1);
      continue;
    }
    written += count;
  }
  return written == bytes;
}

void handleUdp() {
  const int packet_bytes = udp.parsePacket();
  if (packet_bytes <= 0) {
    return;
  }
  if (packet_bytes < static_cast<int>(HEADER_BYTES) ||
      packet_bytes > static_cast<int>(sizeof(udp_buffer))) {
    udp.clear();
    return;
  }
  const int received = udp.read(udp_buffer, sizeof(udp_buffer));
  if (received != packet_bytes) {
    return;
  }

  uint16_t payload_bytes = 0;
  uint32_t request_id = 0;
  const uint8_t request_type = udp_buffer[5];
  if (request_type == TYPE_DISCOVER_REQUEST) {
    if (!validHeader(udp_buffer, TYPE_DISCOVER_REQUEST, 0, payload_bytes,
                     request_id) ||
        packet_bytes != static_cast<int>(HEADER_BYTES)) {
      return;
    }
    writeHeader(udp_buffer, TYPE_DISCOVER_REPLY, 0, request_id);
  } else if (request_type == TYPE_ECHO_REQUEST) {
    if (!validHeader(udp_buffer, TYPE_ECHO_REQUEST, MAX_UDP_PAYLOAD,
                     payload_bytes, request_id) ||
        packet_bytes != static_cast<int>(HEADER_BYTES + payload_bytes)) {
      return;
    }
    writeHeader(udp_buffer, TYPE_ECHO_REPLY, payload_bytes, request_id);
  } else {
    return;
  }

  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.write(udp_buffer, packet_bytes);
  udp.endPacket();
}

void acceptTcpClient() {
  if (tcp_client && tcp_client.connected()) {
    return;
  }
  if (tcp_client) {
    tcp_client.stop();
  }
  tcp_client = tcp_server.accept();
  if (tcp_client) {
    tcp_client.setNoDelay(true);
    Serial.printf("TCP_CLIENT,%s\n", tcp_client.remoteIP().toString().c_str());
  }
}

void handleTcp() {
  acceptTcpClient();
  if (!tcp_client || !tcp_client.connected() ||
      tcp_client.available() < static_cast<int>(HEADER_BYTES)) {
    return;
  }

  uint8_t header[HEADER_BYTES];
  if (!readExact(tcp_client, header, sizeof(header), TCP_READ_TIMEOUT_MS)) {
    tcp_client.stop();
    return;
  }
  uint16_t payload_bytes = 0;
  uint32_t request_id = 0;
  if (!validHeader(header, TYPE_ECHO_REQUEST, MAX_TCP_PAYLOAD, payload_bytes,
                   request_id) ||
      !readExact(tcp_client, tcp_payload, payload_bytes,
                 TCP_READ_TIMEOUT_MS)) {
    tcp_client.stop();
    return;
  }

  writeHeader(header, TYPE_ECHO_REPLY, payload_bytes, request_id);
  if (!writeExact(tcp_client, header, sizeof(header)) ||
      !writeExact(tcp_client, tcp_payload, payload_bytes)) {
    tcp_client.stop();
  }
}

void announceReady() {
  Serial.printf(
      "TRANSPORT_READY,%s,udp=%u,tcp=%u,rssi=%d,free_heap=%u\n",
      WiFi.localIP().toString().c_str(), UDP_PORT, TCP_PORT, WiFi.RSSI(),
      static_cast<unsigned int>(ESP.getFreeHeap()));
}

void startServers() {
  udp.begin(UDP_PORT);
  tcp_server.begin();
  tcp_server.setNoDelay(true);
  announceReady();
}

bool connectWiFi() {
  Serial.printf("WIFI_CONNECTING,%s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const unsigned long deadline = millis() + 30000;
  while (WiFi.status() != WL_CONNECTED &&
         static_cast<long>(deadline - millis()) > 0) {
    delay(250);
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WIFI_FAILED");
    return false;
  }
  startServers();
  return true;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);
  tcp_payload = static_cast<uint8_t *>(malloc(MAX_TCP_PAYLOAD));
  if (!tcp_payload) {
    Serial.println("TRANSPORT_FATAL,tcp_buffer_allocation_failed");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setHostname("esp32-attention-node");
  WiFi.setSleep(false);
  connectWiFi();
  last_wifi_retry = millis();
}

void loop() {
  if (!tcp_payload) {
    delay(1000);
    return;
  }
  if (WiFi.status() == WL_CONNECTED) {
    handleUdp();
    handleTcp();
  } else if (millis() - last_wifi_retry >= WIFI_RETRY_INTERVAL_MS) {
    last_wifi_retry = millis();
    connectWiFi();
  }
  delay(0);
}
