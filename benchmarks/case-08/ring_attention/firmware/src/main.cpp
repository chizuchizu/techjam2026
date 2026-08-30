#include <Arduino.h>
#include <esp_partition.h>
#include <esp_timer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t MAGIC = 0x38474e52u;
constexpr uint8_t VERSION = 1;
constexpr uint8_t REPLY_BIT = 0x80;
constexpr size_t HEADER_BYTES = 20;
constexpr size_t STATUS_BYTES = 8;
constexpr size_t MAX_PAYLOAD = 66560;
constexpr uint16_t MODEL_DIM = 1024;
constexpr uint16_t HEAD_DIM = 256;
constexpr uint16_t MAX_LINEAR_ROWS = 4;
constexpr uint16_t MAX_QUERY_TILE = 16;
constexpr uint16_t MAX_KV_TILE = 32;
constexpr uint16_t FLAG_GELU = 1u << 0;
constexpr uint16_t FLAG_RESIDUAL = 1u << 1;
constexpr uint16_t FLAG_CAUSAL = 1u << 0;
constexpr uint32_t SERIAL_TIMEOUT_MS = 10000;

enum Message : uint8_t {
  HELLO = 1,
  STAGE_BEGIN = 2,
  STAGE_CHUNK = 3,
  STAGE_COMMIT = 4,
  SET_NORM = 5,
  RUN_NORM = 6,
  RUN_LINEAR = 7,
  ATTN_BEGIN = 8,
  ATTN_BLOCK = 9,
  ATTN_END = 10,
  PING = 11,
};

enum Status : uint16_t {
  OK = 0,
  BAD_HEADER = 1,
  BAD_LENGTH = 2,
  BAD_CRC = 3,
  BAD_STATE = 4,
  BAD_SHAPE = 5,
  FLASH_ERROR = 6,
  INTERNAL_ERROR = 7,
};

uint8_t *payload = nullptr;
float linear_output[MAX_LINEAR_ROWS * MODEL_DIM];
int16_t quantized_input[MAX_LINEAR_ROWS * MODEL_DIM];
int16_t weight_row[MODEL_DIM];
float matrix_scales[MODEL_DIM];
float matrix_bias[MODEL_DIM];
float norm_gamma[MODEL_DIM];
float norm_beta[MODEL_DIM];
float attention_query[MAX_QUERY_TILE * HEAD_DIM];
float attention_numerator[MAX_QUERY_TILE * HEAD_DIM];
float attention_maximum[MAX_QUERY_TILE];
float attention_denominator[MAX_QUERY_TILE];

const esp_partition_t *weight_partition = nullptr;
uint32_t staged_bytes = 0;
uint32_t staged_received = 0;
uint32_t staged_crc = 0;
uint16_t staged_rows = 0;
uint16_t staged_cols = 0;
bool matrix_ready = false;
uint16_t norm_dimension = 0;
float norm_epsilon = 1.0e-5f;
bool norm_ready = false;
uint32_t attention_query_begin = 0;
uint16_t attention_query_count = 0;
uint16_t attention_dimension = 0;
bool attention_ready = false;

uint16_t readU16(const uint8_t *source) {
  uint16_t value;
  memcpy(&value, source, sizeof(value));
  return value;
}

uint32_t readU32(const uint8_t *source) {
  uint32_t value;
  memcpy(&value, source, sizeof(value));
  return value;
}

float readFloat(const uint8_t *source) {
  float value;
  memcpy(&value, source, sizeof(value));
  return value;
}

void writeU16(uint8_t *destination, uint16_t value) {
  memcpy(destination, &value, sizeof(value));
}

void writeU32(uint8_t *destination, uint32_t value) {
  memcpy(destination, &value, sizeof(value));
}

void writeFloat(uint8_t *destination, float value) {
  memcpy(destination, &value, sizeof(value));
}

uint32_t crc32Update(uint32_t crc, const uint8_t *data, size_t length) {
  crc = ~crc;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xedb88320u & mask);
    }
  }
  return ~crc;
}

uint32_t crc32(const uint8_t *data, size_t length) {
  return crc32Update(0, data, length);
}

bool readExact(uint8_t *destination, size_t bytes, uint32_t timeout_ms) {
  size_t received = 0;
  const uint32_t started = millis();
  while (received < bytes && millis() - started < timeout_ms) {
    const int available = Serial.available();
    if (available > 0) {
      const size_t count = Serial.readBytes(destination + received,
                                            std::min<size_t>(bytes - received, available));
      received += count;
    } else {
      delay(1);
    }
  }
  return received == bytes;
}

bool writeExact(const uint8_t *source, size_t bytes) {
  size_t written = 0;
  while (written < bytes) {
    const size_t count = Serial.write(source + written, bytes - written);
    if (count == 0) {
      delay(1);
    } else {
      written += count;
    }
  }
  return true;
}

void sendReply(uint8_t request_type, uint32_t request_id, Status status,
               uint32_t elapsed_us, const uint8_t *data = nullptr,
               uint32_t data_bytes = 0) {
  writeU16(payload, static_cast<uint16_t>(status));
  writeU16(payload + 2, 0);
  writeU32(payload + 4, elapsed_us);
  if (data_bytes && data != payload + STATUS_BYTES) {
    memmove(payload + STATUS_BYTES, data, data_bytes);
  }
  const uint32_t total = STATUS_BYTES + data_bytes;
  uint8_t header[HEADER_BYTES];
  writeU32(header, MAGIC);
  header[4] = VERSION;
  header[5] = request_type | REPLY_BIT;
  writeU16(header + 6, 0);
  writeU32(header + 8, request_id);
  writeU32(header + 12, total);
  writeU32(header + 16, crc32(payload, total));
  writeExact(header, sizeof(header));
  writeExact(payload, total);
  Serial.flush();
}

bool partitionCrc(uint32_t bytes, uint32_t &result) {
  uint8_t block[1024];
  uint32_t crc = 0;
  for (uint32_t offset = 0; offset < bytes; offset += sizeof(block)) {
    const size_t count = std::min<uint32_t>(sizeof(block), bytes - offset);
    if (esp_partition_read(weight_partition, offset, block, count) != ESP_OK) {
      return false;
    }
    crc = crc32Update(crc, block, count);
  }
  result = crc;
  return true;
}

float exactGelu(float value) {
  return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
}

Status handleStageBegin(const uint8_t *data, uint32_t bytes) {
  if (bytes != 16 || !weight_partition) return BAD_LENGTH;
  staged_rows = static_cast<uint16_t>(readU32(data));
  staged_cols = static_cast<uint16_t>(readU32(data + 4));
  staged_bytes = readU32(data + 8);
  staged_crc = readU32(data + 12);
  const uint32_t expected = static_cast<uint32_t>(staged_rows) * 8u +
      static_cast<uint32_t>(staged_rows) * staged_cols * sizeof(int16_t);
  if (staged_rows == 0 || staged_rows > MODEL_DIM || staged_cols == 0 ||
      staged_cols > MODEL_DIM || staged_bytes != expected ||
      staged_bytes > weight_partition->size) return BAD_SHAPE;
  const uint32_t erase_bytes = (staged_bytes + 4095u) & ~4095u;
  if (esp_partition_erase_range(weight_partition, 0, erase_bytes) != ESP_OK) {
    return FLASH_ERROR;
  }
  staged_received = 0;
  matrix_ready = false;
  return OK;
}

Status handleStageChunk(const uint8_t *data, uint32_t bytes) {
  if (bytes < 5 || staged_bytes == 0) return BAD_LENGTH;
  const uint32_t offset = readU32(data);
  const uint32_t count = bytes - 4;
  if (offset != staged_received || offset + count > staged_bytes) return BAD_STATE;
  if (esp_partition_write(weight_partition, offset, data + 4, count) != ESP_OK) {
    return FLASH_ERROR;
  }
  staged_received += count;
  return OK;
}

Status handleStageCommit() {
  if (staged_received != staged_bytes) return BAD_STATE;
  uint32_t actual_crc = 0;
  if (!partitionCrc(staged_bytes, actual_crc)) return FLASH_ERROR;
  if (actual_crc != staged_crc) return BAD_CRC;
  const size_t vector_bytes = static_cast<size_t>(staged_rows) * sizeof(float);
  if (esp_partition_read(weight_partition, 0, matrix_scales, vector_bytes) != ESP_OK ||
      esp_partition_read(weight_partition, vector_bytes, matrix_bias, vector_bytes) != ESP_OK) {
    return FLASH_ERROR;
  }
  matrix_ready = true;
  return OK;
}

Status handleSetNorm(const uint8_t *data, uint32_t bytes) {
  if (bytes < 8) return BAD_LENGTH;
  const uint16_t dimension = readU16(data);
  const float epsilon = readFloat(data + 4);
  const size_t expected = 8 + static_cast<size_t>(dimension) * 2 * sizeof(float);
  if (dimension == 0 || dimension > MODEL_DIM || bytes != expected ||
      !isfinite(epsilon) || epsilon <= 0.0f) return BAD_SHAPE;
  memcpy(norm_gamma, data + 8, dimension * sizeof(float));
  memcpy(norm_beta, data + 8 + dimension * sizeof(float), dimension * sizeof(float));
  norm_dimension = dimension;
  norm_epsilon = epsilon;
  norm_ready = true;
  return OK;
}

Status handleRunNorm(const uint8_t *data, uint32_t bytes, uint32_t &output_bytes) {
  if (bytes < 4 || !norm_ready) return BAD_STATE;
  const uint16_t rows = readU16(data);
  const uint16_t dimension = readU16(data + 2);
  const size_t elements = static_cast<size_t>(rows) * dimension;
  if (rows == 0 || rows > MAX_LINEAR_ROWS || dimension != norm_dimension ||
      bytes != 4 + elements * sizeof(float)) return BAD_SHAPE;
  const float *input = reinterpret_cast<const float *>(data + 4);
  for (uint16_t row = 0; row < rows; ++row) {
    const float *source = input + static_cast<size_t>(row) * dimension;
    float mean = 0.0f;
    for (uint16_t feature = 0; feature < dimension; ++feature) mean += source[feature];
    mean /= dimension;
    float variance = 0.0f;
    for (uint16_t feature = 0; feature < dimension; ++feature) {
      const float delta = source[feature] - mean;
      variance += delta * delta;
    }
    const float inverse = 1.0f / sqrtf(variance / dimension + norm_epsilon);
    for (uint16_t feature = 0; feature < dimension; ++feature) {
      linear_output[static_cast<size_t>(row) * dimension + feature] =
          (source[feature] - mean) * inverse * norm_gamma[feature] + norm_beta[feature];
    }
  }
  output_bytes = elements * sizeof(float);
  return OK;
}

Status handleRunLinear(const uint8_t *data, uint32_t bytes, uint32_t &output_bytes) {
  if (bytes < 8 || !matrix_ready) return BAD_STATE;
  const uint16_t rows = readU16(data);
  const uint16_t input_columns = readU16(data + 2);
  const uint16_t output_columns = readU16(data + 4);
  const uint16_t flags = readU16(data + 6);
  const size_t input_elements = static_cast<size_t>(rows) * input_columns;
  const size_t output_elements = static_cast<size_t>(rows) * output_columns;
  const bool residual = (flags & FLAG_RESIDUAL) != 0;
  const size_t expected = 8 + input_elements * sizeof(float) +
      (residual ? output_elements * sizeof(float) : 0);
  if (rows == 0 || rows > MAX_LINEAR_ROWS || input_columns != staged_cols ||
      output_columns != staged_rows || bytes != expected) return BAD_SHAPE;
  const float *input = reinterpret_cast<const float *>(data + 8);
  const float *residual_data = residual ? input + input_elements : nullptr;
  float maximum = 0.0f;
  for (size_t index = 0; index < input_elements; ++index) {
    maximum = std::max(maximum, fabsf(input[index]));
  }
  const float input_scale = maximum > 0.0f ? maximum / 32767.0f : 1.0f;
  const float inverse_scale = 1.0f / input_scale;
  for (size_t index = 0; index < input_elements; ++index) {
    const long value = lrintf(input[index] * inverse_scale);
    quantized_input[index] = static_cast<int16_t>(std::max<long>(-32767, std::min<long>(32767, value)));
  }
  const uint32_t weights_offset = static_cast<uint32_t>(staged_rows) * 8u;
  for (uint16_t output = 0; output < output_columns; ++output) {
    const uint32_t offset = weights_offset +
        static_cast<uint32_t>(output) * input_columns * sizeof(int16_t);
    if (esp_partition_read(weight_partition, offset, weight_row,
                           input_columns * sizeof(int16_t)) != ESP_OK) return FLASH_ERROR;
    for (uint16_t row = 0; row < rows; ++row) {
      const int16_t *source = quantized_input + static_cast<size_t>(row) * input_columns;
      int64_t accumulator = 0;
      for (uint16_t input_index = 0; input_index < input_columns; ++input_index) {
        accumulator += static_cast<int32_t>(source[input_index]) * weight_row[input_index];
      }
      float value = static_cast<float>(accumulator) * input_scale * matrix_scales[output] +
                    matrix_bias[output];
      if (flags & FLAG_GELU) value = exactGelu(value);
      const size_t result_index = static_cast<size_t>(row) * output_columns + output;
      if (residual) value += residual_data[result_index];
      linear_output[result_index] = value;
    }
  }
  output_bytes = output_elements * sizeof(float);
  return OK;
}

Status handleAttentionBegin(const uint8_t *data, uint32_t bytes) {
  if (bytes < 8) return BAD_LENGTH;
  const uint32_t query_begin = readU32(data);
  const uint16_t query_count = readU16(data + 4);
  const uint16_t dimension = readU16(data + 6);
  const size_t elements = static_cast<size_t>(query_count) * dimension;
  if (query_count == 0 || query_count > MAX_QUERY_TILE || dimension != HEAD_DIM ||
      bytes != 8 + elements * sizeof(float)) return BAD_SHAPE;
  memcpy(attention_query, data + 8, elements * sizeof(float));
  memset(attention_numerator, 0, elements * sizeof(float));
  for (uint16_t query = 0; query < query_count; ++query) {
    attention_maximum[query] = -INFINITY;
    attention_denominator[query] = 0.0f;
  }
  attention_query_begin = query_begin;
  attention_query_count = query_count;
  attention_dimension = dimension;
  attention_ready = true;
  return OK;
}

Status handleAttentionBlock(const uint8_t *data, uint32_t bytes) {
  if (bytes < 12 || !attention_ready) return BAD_STATE;
  const uint32_t kv_begin = readU32(data);
  const uint16_t kv_count = readU16(data + 4);
  const uint16_t dimension = readU16(data + 6);
  const uint16_t flags = readU16(data + 8);
  const size_t elements = static_cast<size_t>(kv_count) * dimension;
  if (kv_count == 0 || kv_count > MAX_KV_TILE || dimension != attention_dimension ||
      bytes != 12 + elements * 2 * sizeof(float)) return BAD_SHAPE;
  const float *key = reinterpret_cast<const float *>(data + 12);
  const float *value = key + elements;
  const float score_scale = 1.0f / sqrtf(static_cast<float>(dimension));
  for (uint16_t query_index = 0; query_index < attention_query_count; ++query_index) {
    const uint32_t global_query = attention_query_begin + query_index;
    const float *query = attention_query + static_cast<size_t>(query_index) * dimension;
    float *numerator = attention_numerator + static_cast<size_t>(query_index) * dimension;
    for (uint16_t key_index = 0; key_index < kv_count; ++key_index) {
      const uint32_t global_key = kv_begin + key_index;
      if ((flags & FLAG_CAUSAL) && global_key > global_query) continue;
      const float *key_row = key + static_cast<size_t>(key_index) * dimension;
      const float *value_row = value + static_cast<size_t>(key_index) * dimension;
      float dot = 0.0f;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        dot += query[feature] * key_row[feature];
      }
      const float score = dot * score_scale;
      const float old_maximum = attention_maximum[query_index];
      const float new_maximum = std::max(old_maximum, score);
      const float old_scale = isfinite(old_maximum) ? expf(old_maximum - new_maximum) : 0.0f;
      const float new_scale = expf(score - new_maximum);
      attention_denominator[query_index] =
          attention_denominator[query_index] * old_scale + new_scale;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        numerator[feature] = numerator[feature] * old_scale + new_scale * value_row[feature];
      }
      attention_maximum[query_index] = new_maximum;
    }
  }
  return OK;
}

Status handleAttentionEnd(uint32_t &output_bytes) {
  if (!attention_ready) return BAD_STATE;
  const size_t elements = static_cast<size_t>(attention_query_count) * attention_dimension;
  for (uint16_t query = 0; query < attention_query_count; ++query) {
    const float denominator = attention_denominator[query];
    if (!(denominator > 0.0f) || !isfinite(denominator)) return INTERNAL_ERROR;
    for (uint16_t feature = 0; feature < attention_dimension; ++feature) {
      const size_t index = static_cast<size_t>(query) * attention_dimension + feature;
      linear_output[index] = attention_numerator[index] / denominator;
    }
  }
  attention_ready = false;
  output_bytes = elements * sizeof(float);
  return OK;
}

void handleFrame(uint8_t request_type, uint32_t request_id, const uint8_t *data,
                 uint32_t bytes) {
  const int64_t started = esp_timer_get_time();
  Status status = OK;
  uint32_t output_bytes = 0;
  const uint8_t *output = nullptr;
  switch (request_type) {
    case HELLO: {
      const char *identity = "case8-ring-worker,c3,S=128,D=1024,H=4,HD=256,q=16,kv=32";
      output = reinterpret_cast<const uint8_t *>(identity);
      output_bytes = strlen(identity);
      break;
    }
    case STAGE_BEGIN: status = handleStageBegin(data, bytes); break;
    case STAGE_CHUNK: status = handleStageChunk(data, bytes); break;
    case STAGE_COMMIT: status = bytes == 0 ? handleStageCommit() : BAD_LENGTH; break;
    case SET_NORM: status = handleSetNorm(data, bytes); break;
    case RUN_NORM:
      status = handleRunNorm(data, bytes, output_bytes);
      output = reinterpret_cast<const uint8_t *>(linear_output);
      break;
    case RUN_LINEAR:
      status = handleRunLinear(data, bytes, output_bytes);
      output = reinterpret_cast<const uint8_t *>(linear_output);
      break;
    case ATTN_BEGIN: status = handleAttentionBegin(data, bytes); break;
    case ATTN_BLOCK: status = handleAttentionBlock(data, bytes); break;
    case ATTN_END:
      status = bytes == 0 ? handleAttentionEnd(output_bytes) : BAD_LENGTH;
      output = reinterpret_cast<const uint8_t *>(linear_output);
      break;
    case PING: output = data; output_bytes = bytes; break;
    default: status = BAD_HEADER; break;
  }
  const uint32_t elapsed = static_cast<uint32_t>(esp_timer_get_time() - started);
  sendReply(request_type, request_id, status, elapsed, output, output_bytes);
}

}  // namespace

void setup() {
  Serial.begin(921600);
  delay(500);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);
  weight_partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "weights");
  payload = static_cast<uint8_t *>(malloc(MAX_PAYLOAD + STATUS_BYTES));
  if (!payload || !weight_partition) {
    Serial.println("CASE8_RING_FATAL,allocation_or_partition");
    return;
  }
  Serial.printf("CASE8_RING_READY,model=%s,heap=%u,flash_scratch=%u\n",
                ESP.getChipModel(), static_cast<unsigned>(ESP.getFreeHeap()),
                static_cast<unsigned>(weight_partition->size));
}

void loop() {
  if (!payload || !weight_partition) {
    delay(1000);
    return;
  }
  if (Serial.available() < static_cast<int>(HEADER_BYTES)) {
    delay(1);
    return;
  }
  uint8_t header[HEADER_BYTES];
  if (!readExact(header, sizeof(header), SERIAL_TIMEOUT_MS)) return;
  const uint32_t magic = readU32(header);
  const uint8_t version = header[4];
  const uint8_t request_type = header[5];
  const uint32_t request_id = readU32(header + 8);
  const uint32_t bytes = readU32(header + 12);
  const uint32_t expected_crc = readU32(header + 16);
  if (magic != MAGIC || version != VERSION || (request_type & REPLY_BIT) ||
      bytes > MAX_PAYLOAD) {
    sendReply(request_type, request_id, BAD_HEADER, 0);
    return;
  }
  if (!readExact(payload, bytes, SERIAL_TIMEOUT_MS)) {
    sendReply(request_type, request_id, BAD_LENGTH, 0);
    return;
  }
  if (crc32(payload, bytes) != expected_crc) {
    sendReply(request_type, request_id, BAD_CRC, 0);
    return;
  }
  handleFrame(request_type, request_id, payload, bytes);
}
