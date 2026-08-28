#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <cfloat>
#include <esp_timer.h>

#include "secrets.h"

namespace {

constexpr uint32_t MAGIC = 0x45535033u;  // "ESP3"
constexpr uint8_t PROTOCOL_VERSION = 1;
constexpr uint8_t TYPE_ECHO_REQUEST = 1;
constexpr uint8_t TYPE_ECHO_REPLY = 2;
constexpr uint8_t TYPE_DISCOVER_REQUEST = 3;
constexpr uint8_t TYPE_DISCOVER_REPLY = 4;
constexpr uint8_t TYPE_HEAD_TASK = 5;
constexpr uint8_t TYPE_HEAD_RESULT = 6;
constexpr uint8_t TYPE_KV_SHARD_TASK = 7;
constexpr uint8_t TYPE_KV_SHARD_RESULT = 8;
constexpr uint8_t TYPE_CAPABILITIES_REQUEST = 9;
constexpr uint8_t TYPE_CAPABILITIES_REPLY = 10;
constexpr uint32_t CAPABILITY_HEAD_UDP = 1u << 0;
constexpr uint32_t CAPABILITY_HEAD_TCP = 1u << 1;
constexpr uint32_t CAPABILITY_KV_SHARD_UDP = 1u << 2;
constexpr uint16_t UDP_PORT = 4210;
constexpr uint16_t TCP_PORT = 4211;
constexpr size_t HEADER_BYTES = 12;
constexpr size_t MAX_UDP_PAYLOAD = 1400;
constexpr size_t MAX_TCP_PAYLOAD = 32768;
constexpr uint16_t MAX_HEAD_SEQUENCE = 128;
constexpr uint16_t MAX_HEAD_DIMENSION = 32;
constexpr uint16_t MAX_HEAD_ELEMENTS =
    MAX_HEAD_SEQUENCE * MAX_HEAD_DIMENSION;
constexpr uint16_t MAX_HEAD_TILE = 32;
constexpr size_t HEAD_TASK_FIXED_BYTES = 20;
constexpr size_t HEAD_RESULT_FIXED_BYTES = 16;
constexpr size_t KV_SHARD_TASK_FIXED_BYTES = 24;
constexpr size_t KV_SHARD_RESULT_FIXED_BYTES = 20;
constexpr size_t CAPABILITIES_FIXED_BYTES = 20;
constexpr unsigned long WIFI_RETRY_INTERVAL_MS = 10000;
constexpr unsigned long TCP_READ_TIMEOUT_MS = 3000;

WiFiUDP udp;
WiFiServer tcp_server(TCP_PORT);
WiFiClient tcp_client;
uint8_t udp_buffer[HEADER_BYTES + MAX_UDP_PAYLOAD];
uint8_t *tcp_payload = nullptr;
int8_t head_query[MAX_HEAD_ELEMENTS];
int8_t head_key[MAX_HEAD_ELEMENTS];
int16_t head_value[MAX_HEAD_ELEMENTS];
float head_output[MAX_HEAD_ELEMENTS];
float head_scratch_scores[MAX_HEAD_TILE];
float head_scratch_value[MAX_HEAD_DIMENSION];
uint16_t head_scratch_indices[MAX_HEAD_TILE];
uint8_t head_valid_mask[(MAX_HEAD_SEQUENCE + 7) / 8];
float shard_maximum[MAX_HEAD_SEQUENCE];
float shard_sum[MAX_HEAD_SEQUENCE];
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

float readFloat(const uint8_t *source) {
  const uint32_t bits = readU32(source);
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

void writeFloat(uint8_t *destination, float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  writeU32(destination, bits);
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

bool headTokenIsValid(uint16_t token) {
  return (head_valid_mask[token / 8] & (1u << (token % 8))) != 0;
}

void computeHeadAttention(uint16_t sequence,
                          uint16_t dimension,
                          uint16_t tile_size,
                          bool causal,
                          float query_scale,
                          float key_scale,
                          float value_scale) {
  const float score_scale =
      query_scale * key_scale / sqrtf(static_cast<float>(dimension));
  memset(head_output, 0,
         static_cast<size_t>(sequence) * dimension * sizeof(float));

  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    if (!headTokenIsValid(query_index)) {
      continue;
    }
    const int8_t *query_row =
        head_query + static_cast<size_t>(query_index) * dimension;
    float *output_row =
        head_output + static_cast<size_t>(query_index) * dimension;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    bool has_values = false;

    for (uint16_t tile_begin = 0; tile_begin < sequence;
         tile_begin += tile_size) {
      const uint16_t tile_end =
          static_cast<uint16_t>(tile_begin + tile_size < sequence
                                    ? tile_begin + tile_size
                                    : sequence);
      uint16_t tile_count = 0;
      float tile_max = -FLT_MAX;
      for (uint16_t key_index = tile_begin; key_index < tile_end; ++key_index) {
        if (!headTokenIsValid(key_index) ||
            (causal && key_index > query_index)) {
          continue;
        }
        const int8_t *key_row =
            head_key + static_cast<size_t>(key_index) * dimension;
        int32_t dot = 0;
        for (uint16_t feature = 0; feature < dimension; ++feature) {
          dot += static_cast<int32_t>(query_row[feature]) * key_row[feature];
        }
        const float score = static_cast<float>(dot) * score_scale;
        head_scratch_scores[tile_count] = score;
        head_scratch_indices[tile_count] = key_index;
        tile_max = fmaxf(tile_max, score);
        ++tile_count;
      }
      if (tile_count == 0) {
        continue;
      }

      memset(head_scratch_value, 0,
             static_cast<size_t>(dimension) * sizeof(float));
      float tile_sum = 0.0f;
      for (uint16_t offset = 0; offset < tile_count; ++offset) {
        const float weight = expf(head_scratch_scores[offset] - tile_max);
        tile_sum += weight;
        const int16_t *value_row =
            head_value +
            static_cast<size_t>(head_scratch_indices[offset]) * dimension;
        for (uint16_t feature = 0; feature < dimension; ++feature) {
          head_scratch_value[feature] += weight * value_row[feature];
        }
      }

      if (!has_values) {
        memcpy(output_row, head_scratch_value,
               static_cast<size_t>(dimension) * sizeof(float));
        running_max = tile_max;
        running_sum = tile_sum;
        has_values = true;
        continue;
      }

      const float merged_max = fmaxf(running_max, tile_max);
      const float old_scale = expf(running_max - merged_max);
      const float tile_scale = expf(tile_max - merged_max);
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        output_row[feature] = output_row[feature] * old_scale +
                              head_scratch_value[feature] * tile_scale;
      }
      running_sum = running_sum * old_scale + tile_sum * tile_scale;
      running_max = merged_max;
    }

    if (has_values) {
      const float output_scale = value_scale / running_sum;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        output_row[feature] *= output_scale;
      }
    }
  }
}

int handleHeadTaskPayload(const uint8_t *payload,
                          size_t payload_bytes,
                          uint8_t *response,
                          size_t response_capacity) {
  const uint16_t sequence = readU16(payload);
  const uint16_t dimension = readU16(payload + 2);
  const bool causal = (payload[4] & 1u) != 0;
  const uint16_t tile_size = payload[5];
  const uint16_t mask_bytes = readU16(payload + 6);
  const float query_scale = readFloat(payload + 8);
  const float key_scale = readFloat(payload + 12);
  const float value_scale = readFloat(payload + 16);

  if (sequence == 0 || sequence > MAX_HEAD_SEQUENCE || dimension == 0 ||
      dimension > MAX_HEAD_DIMENSION || tile_size == 0 ||
      tile_size > MAX_HEAD_TILE ||
      mask_bytes != (sequence + 7) / 8 || !isfinite(query_scale) ||
      !isfinite(key_scale) || !isfinite(value_scale) || query_scale <= 0.0f ||
      key_scale <= 0.0f || value_scale <= 0.0f) {
    return 0;
  }
  const size_t elements = static_cast<size_t>(sequence) * dimension;
  const size_t expected_payload = HEAD_TASK_FIXED_BYTES + mask_bytes +
                                  elements * 2 + elements * sizeof(int16_t);
  if (payload_bytes != expected_payload) {
    return 0;
  }

  const int64_t decode_start = esp_timer_get_time();
  const uint8_t *cursor = payload + HEAD_TASK_FIXED_BYTES;
  memcpy(head_valid_mask, cursor, mask_bytes);
  cursor += mask_bytes;
  memcpy(head_query, cursor, elements);
  cursor += elements;
  memcpy(head_key, cursor, elements);
  cursor += elements;
  for (size_t index = 0; index < elements; ++index) {
    head_value[index] = static_cast<int16_t>(readU16(cursor + index * 2));
  }
  const uint32_t decode_us =
      static_cast<uint32_t>(esp_timer_get_time() - decode_start);

  const int64_t compute_start = esp_timer_get_time();
  computeHeadAttention(sequence, dimension, tile_size, causal, query_scale,
                       key_scale, value_scale);
  const uint32_t compute_us =
      static_cast<uint32_t>(esp_timer_get_time() - compute_start);

  const uint16_t response_payload_bytes =
      static_cast<uint16_t>(HEAD_RESULT_FIXED_BYTES +
                            elements * sizeof(float));
  if (response_payload_bytes > response_capacity) {
    return 0;
  }
  writeU16(response, sequence);
  writeU16(response + 2, dimension);
  response[4] = causal ? 1 : 0;
  response[5] = 0;  // status
  writeU16(response + 6, static_cast<uint16_t>(elements));
  writeU32(response + 8, decode_us);
  writeU32(response + 12, compute_us);
  for (size_t index = 0; index < elements; ++index) {
    writeFloat(response + HEAD_RESULT_FIXED_BYTES + index * sizeof(float),
               head_output[index]);
  }
  return response_payload_bytes;
}

void computeKvShardStatistics(uint16_t sequence,
                              uint16_t dimension,
                              uint16_t shard_begin,
                              uint16_t shard_end,
                              bool causal,
                              float query_scale,
                              float key_scale) {
  const float score_scale =
      query_scale * key_scale / sqrtf(static_cast<float>(dimension));
  const uint16_t shard_keys = shard_end - shard_begin;
  memset(head_output, 0,
         static_cast<size_t>(sequence) * dimension * sizeof(float));

  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    shard_maximum[query_index] = -INFINITY;
    shard_sum[query_index] = 0.0f;
    if (!headTokenIsValid(query_index)) {
      continue;
    }
    const int8_t *query_row =
        head_query + static_cast<size_t>(query_index) * dimension;
    bool has_keys = false;
    float local_maximum = -FLT_MAX;
    for (uint16_t local_key = 0; local_key < shard_keys; ++local_key) {
      const uint16_t key_index = shard_begin + local_key;
      if (!headTokenIsValid(key_index) ||
          (causal && key_index > query_index)) {
        continue;
      }
      const int8_t *key_row =
          head_key + static_cast<size_t>(local_key) * dimension;
      int32_t dot = 0;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        dot += static_cast<int32_t>(query_row[feature]) * key_row[feature];
      }
      local_maximum =
          fmaxf(local_maximum, static_cast<float>(dot) * score_scale);
      has_keys = true;
    }
    if (!has_keys) {
      continue;
    }

    shard_maximum[query_index] = local_maximum;
    float *numerator =
        head_output + static_cast<size_t>(query_index) * dimension;
    float local_sum = 0.0f;
    for (uint16_t local_key = 0; local_key < shard_keys; ++local_key) {
      const uint16_t key_index = shard_begin + local_key;
      if (!headTokenIsValid(key_index) ||
          (causal && key_index > query_index)) {
        continue;
      }
      const int8_t *key_row =
          head_key + static_cast<size_t>(local_key) * dimension;
      int32_t dot = 0;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        dot += static_cast<int32_t>(query_row[feature]) * key_row[feature];
      }
      const float score = static_cast<float>(dot) * score_scale;
      const float weight = expf(score - local_maximum);
      local_sum += weight;
      const int16_t *value_row =
          head_value + static_cast<size_t>(local_key) * dimension;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        numerator[feature] += weight * value_row[feature];
      }
    }
    shard_sum[query_index] = local_sum;
  }
}

int handleKvShardTask(int packet_bytes, uint32_t request_id) {
  const uint8_t *payload = udp_buffer + HEADER_BYTES;
  const uint16_t sequence = readU16(payload);
  const uint16_t dimension = readU16(payload + 2);
  const bool causal = (payload[4] & 1u) != 0;
  const uint16_t mask_bytes = readU16(payload + 6);
  const uint16_t shard_begin = readU16(payload + 8);
  const uint16_t shard_end = readU16(payload + 10);
  const float query_scale = readFloat(payload + 12);
  const float key_scale = readFloat(payload + 16);
  const float value_scale = readFloat(payload + 20);

  if (sequence == 0 || sequence > MAX_HEAD_SEQUENCE || dimension == 0 ||
      dimension > MAX_HEAD_DIMENSION || shard_begin >= shard_end ||
      shard_end > sequence || mask_bytes != (sequence + 7) / 8 ||
      !isfinite(query_scale) || !isfinite(key_scale) ||
      !isfinite(value_scale) || query_scale <= 0.0f || key_scale <= 0.0f ||
      value_scale <= 0.0f) {
    return 0;
  }
  const size_t query_elements = static_cast<size_t>(sequence) * dimension;
  const size_t shard_elements =
      static_cast<size_t>(shard_end - shard_begin) * dimension;
  const size_t expected_payload =
      KV_SHARD_TASK_FIXED_BYTES + mask_bytes + query_elements +
      shard_elements + shard_elements * sizeof(int16_t);
  if (packet_bytes != static_cast<int>(HEADER_BYTES + expected_payload)) {
    return 0;
  }

  const int64_t decode_start = esp_timer_get_time();
  const uint8_t *cursor = payload + KV_SHARD_TASK_FIXED_BYTES;
  memcpy(head_valid_mask, cursor, mask_bytes);
  cursor += mask_bytes;
  memcpy(head_query, cursor, query_elements);
  cursor += query_elements;
  memcpy(head_key, cursor, shard_elements);
  cursor += shard_elements;
  for (size_t index = 0; index < shard_elements; ++index) {
    head_value[index] = static_cast<int16_t>(readU16(cursor + index * 2));
  }
  const uint32_t decode_us =
      static_cast<uint32_t>(esp_timer_get_time() - decode_start);

  const int64_t compute_start = esp_timer_get_time();
  computeKvShardStatistics(sequence, dimension, shard_begin, shard_end, causal,
                           query_scale, key_scale);
  const uint32_t compute_us =
      static_cast<uint32_t>(esp_timer_get_time() - compute_start);

  const size_t statistics_floats =
      static_cast<size_t>(sequence) * (dimension + 2);
  const uint16_t response_payload_bytes = static_cast<uint16_t>(
      KV_SHARD_RESULT_FIXED_BYTES + statistics_floats * sizeof(float));
  writeHeader(udp_buffer, TYPE_KV_SHARD_RESULT, response_payload_bytes,
              request_id);
  uint8_t *response = udp_buffer + HEADER_BYTES;
  writeU16(response, sequence);
  writeU16(response + 2, dimension);
  response[4] = causal ? 1 : 0;
  response[5] = 0;
  writeU16(response + 6, shard_begin);
  writeU16(response + 8, shard_end);
  writeU16(response + 10, sequence);
  writeU32(response + 12, decode_us);
  writeU32(response + 16, compute_us);
  uint8_t *statistics = response + KV_SHARD_RESULT_FIXED_BYTES;
  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    writeFloat(statistics, shard_maximum[query_index]);
    writeFloat(statistics + 4, shard_sum[query_index]);
    statistics += 8;
    const float *numerator =
        head_output + static_cast<size_t>(query_index) * dimension;
    for (uint16_t feature = 0; feature < dimension; ++feature) {
      writeFloat(statistics, numerator[feature]);
      statistics += 4;
    }
  }
  return HEADER_BYTES + response_payload_bytes;
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
  int reply_bytes = packet_bytes;
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
  } else if (request_type == TYPE_HEAD_TASK) {
    if (!validHeader(udp_buffer, TYPE_HEAD_TASK, MAX_UDP_PAYLOAD,
                     payload_bytes, request_id) ||
        packet_bytes != static_cast<int>(HEADER_BYTES + payload_bytes)) {
      return;
    }
    const int response_payload_bytes = handleHeadTaskPayload(
        udp_buffer + HEADER_BYTES, payload_bytes, udp_buffer + HEADER_BYTES,
        MAX_UDP_PAYLOAD);
    if (response_payload_bytes == 0) {
      return;
    }
    writeHeader(udp_buffer, TYPE_HEAD_RESULT, response_payload_bytes,
                request_id);
    reply_bytes = HEADER_BYTES + response_payload_bytes;
  } else if (request_type == TYPE_KV_SHARD_TASK) {
    if (!validHeader(udp_buffer, TYPE_KV_SHARD_TASK, MAX_UDP_PAYLOAD,
                     payload_bytes, request_id)) {
      return;
    }
    reply_bytes = handleKvShardTask(packet_bytes, request_id);
    if (reply_bytes == 0) {
      return;
    }
  } else if (request_type == TYPE_CAPABILITIES_REQUEST) {
    if (!validHeader(udp_buffer, TYPE_CAPABILITIES_REQUEST, 0, payload_bytes,
                     request_id) ||
        packet_bytes != static_cast<int>(HEADER_BYTES)) {
      return;
    }
    const char *model = ESP.getChipModel();
    const size_t model_length = strnlen(model, 31);
    const uint16_t response_payload_bytes =
        static_cast<uint16_t>(CAPABILITIES_FIXED_BYTES + model_length);
    writeHeader(udp_buffer, TYPE_CAPABILITIES_REPLY, response_payload_bytes,
                request_id);
    uint8_t *response = udp_buffer + HEADER_BYTES;
    writeU32(response, CAPABILITY_HEAD_UDP | CAPABILITY_HEAD_TCP |
                           CAPABILITY_KV_SHARD_UDP);
    writeU16(response + 4, MAX_HEAD_SEQUENCE);
    writeU16(response + 6, MAX_HEAD_DIMENSION);
    writeU16(response + 8, MAX_UDP_PAYLOAD);
    writeU16(response + 10, MAX_TCP_PAYLOAD);
    writeU32(response + 12, ESP.getFreeHeap());
    writeU16(response + 16, ESP.getCpuFreqMHz());
    response[18] = ESP.getChipCores();
    response[19] = static_cast<uint8_t>(model_length);
    memcpy(response + CAPABILITIES_FIXED_BYTES, model, model_length);
    reply_bytes = HEADER_BYTES + response_payload_bytes;
  } else {
    return;
  }

  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.write(udp_buffer, reply_bytes);
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
  const uint8_t request_type = header[5];
  const bool valid_echo =
      request_type == TYPE_ECHO_REQUEST &&
      validHeader(header, TYPE_ECHO_REQUEST, MAX_TCP_PAYLOAD, payload_bytes,
                  request_id);
  const bool valid_head =
      request_type == TYPE_HEAD_TASK &&
      validHeader(header, TYPE_HEAD_TASK, MAX_TCP_PAYLOAD, payload_bytes,
                  request_id);
  if ((!valid_echo && !valid_head) ||
      !readExact(tcp_client, tcp_payload, payload_bytes,
                 TCP_READ_TIMEOUT_MS)) {
    tcp_client.stop();
    return;
  }

  if (valid_echo) {
    writeHeader(header, TYPE_ECHO_REPLY, payload_bytes, request_id);
  } else {
    const int response_payload_bytes = handleHeadTaskPayload(
        tcp_payload, payload_bytes, tcp_payload, MAX_TCP_PAYLOAD);
    if (response_payload_bytes == 0) {
      tcp_client.stop();
      return;
    }
    payload_bytes = static_cast<uint16_t>(response_payload_bytes);
    writeHeader(header, TYPE_HEAD_RESULT, payload_bytes, request_id);
  }
  if (!writeExact(tcp_client, header, sizeof(header)) ||
      !writeExact(tcp_client, tcp_payload, payload_bytes)) {
    tcp_client.stop();
  }
}

void announceReady() {
  Serial.printf(
      "TRANSPORT_READY,%s,model=%s,cores=%u,mhz=%u,udp=%u,tcp=%u,rssi=%d,free_heap=%u\n",
      WiFi.localIP().toString().c_str(), ESP.getChipModel(),
      ESP.getChipCores(), ESP.getCpuFreqMHz(), UDP_PORT,
      TCP_PORT, WiFi.RSSI(), static_cast<unsigned int>(ESP.getFreeHeap()));
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
