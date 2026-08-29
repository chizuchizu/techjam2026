#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>

#include "attention_kernels.h"
#include "end_to_end_attention.h"

namespace {

constexpr uint16_t TILE_SIZE = 8;
constexpr float RELATIVE_TOLERANCE = 0.02f;
constexpr float ABSOLUTE_TOLERANCE = 0.002f;
constexpr uint8_t MAX_TIMING_SAMPLES = 7;

constexpr EndToEndConfig END_TO_END_CONFIG = {16, 32, 4, TILE_SIZE};

const AttentionShape SHAPES[] = {
    {8, 8, false},
    {16, 16, false},
    {32, 16, false},
    {32, 32, false},
    {64, 32, false},
    {64, 32, true},
    {64, 64, false},
    {96, 32, false},
    {128, 32, false},
};

enum class Kernel : uint8_t {
  Materialized,
  TiledExact,
  TiledFastExp,
  Int8Materialized,
  Int8TiledExact,
  Int8QKInt16VMaterialized,
  Int8QKInt16VTiledExact,
};

struct Buffers {
  float *query = nullptr;
  float *key = nullptr;
  float *value = nullptr;
  float *reference = nullptr;
  float *candidate = nullptr;
  float *scores = nullptr;
  float *scratch_scores = nullptr;
  float *scratch_value = nullptr;
  int8_t *query_int8 = nullptr;
  int8_t *key_int8 = nullptr;
  int8_t *value_int8 = nullptr;
  int16_t *value_int16 = nullptr;
  QuantizationScales scales = {1.0f, 1.0f, 1.0f};
  QuantizationScales scales_int16 = {1.0f, 1.0f, 1.0f};
};

uint32_t random_state = 0xC001D00Du;

uint32_t nextRandom() {
  random_state = random_state * 1664525u + 1013904223u;
  return random_state;
}

float randomFloat() {
  const int32_t centered = static_cast<int32_t>((nextRandom() >> 8) & 0xFFFFu) -
                           32768;
  return static_cast<float>(centered) / 65536.0f;
}

void fillInputs(Buffers &buffers, size_t elements) {
  random_state = 0xC001D00Du;
  for (size_t index = 0; index < elements; ++index) {
    buffers.query[index] = randomFloat();
    buffers.key[index] = randomFloat();
    buffers.value[index] = randomFloat() * 2.0f;
  }
}

float quantizeSymmetric(const float *source,
                        int8_t *destination,
                        size_t elements) {
  float maximum = 0.0f;
  for (size_t index = 0; index < elements; ++index) {
    const float magnitude = fabsf(source[index]);
    if (magnitude > maximum) {
      maximum = magnitude;
    }
  }

  const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
  const float inverse_scale = 1.0f / scale;
  for (size_t index = 0; index < elements; ++index) {
    int32_t quantized =
        static_cast<int32_t>(lroundf(source[index] * inverse_scale));
    if (quantized > 127) {
      quantized = 127;
    } else if (quantized < -127) {
      quantized = -127;
    }
    destination[index] = static_cast<int8_t>(quantized);
  }
  return scale;
}

float quantizeSymmetricInt16(const float *source,
                             int16_t *destination,
                             size_t elements) {
  float maximum = 0.0f;
  for (size_t index = 0; index < elements; ++index) {
    const float magnitude = fabsf(source[index]);
    if (magnitude > maximum) {
      maximum = magnitude;
    }
  }

  const float scale = maximum > 0.0f ? maximum / 32767.0f : 1.0f;
  const float inverse_scale = 1.0f / scale;
  for (size_t index = 0; index < elements; ++index) {
    int32_t quantized =
        static_cast<int32_t>(lroundf(source[index] * inverse_scale));
    if (quantized > 32767) {
      quantized = 32767;
    } else if (quantized < -32767) {
      quantized = -32767;
    }
    destination[index] = static_cast<int16_t>(quantized);
  }
  return scale;
}

void quantizeInputs(Buffers &buffers, size_t elements) {
  buffers.scales.query =
      quantizeSymmetric(buffers.query, buffers.query_int8, elements);
  buffers.scales.key =
      quantizeSymmetric(buffers.key, buffers.key_int8, elements);
  buffers.scales.value =
      quantizeSymmetric(buffers.value, buffers.value_int8, elements);
  buffers.scales_int16.query = buffers.scales.query;
  buffers.scales_int16.key = buffers.scales.key;
  buffers.scales_int16.value =
      quantizeSymmetricInt16(buffers.value, buffers.value_int16, elements);
}

void releaseBuffers(Buffers &buffers) {
  free(buffers.query);
  free(buffers.key);
  free(buffers.value);
  free(buffers.reference);
  free(buffers.candidate);
  free(buffers.scores);
  free(buffers.scratch_scores);
  free(buffers.scratch_value);
  free(buffers.query_int8);
  free(buffers.key_int8);
  free(buffers.value_int8);
  free(buffers.value_int16);
  buffers = Buffers{};
}

bool allocateBuffers(const AttentionShape &shape, Buffers &buffers) {
  const size_t matrix_elements =
      static_cast<size_t>(shape.sequence) * shape.head_dimension;
  const size_t matrix_bytes = matrix_elements * sizeof(float);
  buffers.query = static_cast<float *>(malloc(matrix_bytes));
  buffers.key = static_cast<float *>(malloc(matrix_bytes));
  buffers.value = static_cast<float *>(malloc(matrix_bytes));
  buffers.reference = static_cast<float *>(malloc(matrix_bytes));
  buffers.candidate = static_cast<float *>(malloc(matrix_bytes));
  buffers.scores = static_cast<float *>(malloc(materializedWorkspaceBytes(shape)));
  buffers.scratch_scores =
      static_cast<float *>(malloc(TILE_SIZE * sizeof(float)));
  buffers.scratch_value =
      static_cast<float *>(malloc(shape.head_dimension * sizeof(float)));
  buffers.query_int8 = static_cast<int8_t *>(malloc(matrix_elements));
  buffers.key_int8 = static_cast<int8_t *>(malloc(matrix_elements));
  buffers.value_int8 = static_cast<int8_t *>(malloc(matrix_elements));
  buffers.value_int16 =
      static_cast<int16_t *>(malloc(matrix_elements * sizeof(int16_t)));

  const bool allocated = buffers.query && buffers.key && buffers.value &&
                         buffers.reference && buffers.candidate &&
                         buffers.scores && buffers.scratch_scores &&
                         buffers.scratch_value && buffers.query_int8 &&
                         buffers.key_int8 && buffers.value_int8 &&
                         buffers.value_int16;
  if (!allocated) {
    releaseBuffers(buffers);
  }
  return allocated;
}

void runKernel(Kernel kernel,
               const AttentionShape &shape,
               Buffers &buffers,
               float *output) {
  switch (kernel) {
    case Kernel::Materialized:
      attentionMaterializedReference(buffers.query, buffers.key, buffers.value,
                                     output, buffers.scores, shape);
      break;
    case Kernel::TiledExact:
      attentionTiledOnlineExact(buffers.query, buffers.key, buffers.value,
                                output, buffers.scratch_scores,
                                buffers.scratch_value, TILE_SIZE, shape);
      break;
    case Kernel::TiledFastExp:
      attentionTiledOnlineFastExp(buffers.query, buffers.key, buffers.value,
                                  output, buffers.scratch_scores,
                                  buffers.scratch_value, TILE_SIZE, shape);
      break;
    case Kernel::Int8Materialized:
      attentionMaterializedInt8(
          buffers.query_int8, buffers.key_int8, buffers.value_int8,
          buffers.scales, output, buffers.scores, shape);
      break;
    case Kernel::Int8TiledExact:
      attentionTiledOnlineInt8(
          buffers.query_int8, buffers.key_int8, buffers.value_int8,
          buffers.scales, output, buffers.scratch_scores,
          buffers.scratch_value, TILE_SIZE, shape);
      break;
    case Kernel::Int8QKInt16VMaterialized:
      attentionMaterializedInt8QKInt16V(
          buffers.query_int8, buffers.key_int8, buffers.value_int16,
          buffers.scales_int16, output, buffers.scores, shape);
      break;
    case Kernel::Int8QKInt16VTiledExact:
      attentionTiledOnlineInt8QKInt16V(
          buffers.query_int8, buffers.key_int8, buffers.value_int16,
          buffers.scales_int16, output, buffers.scratch_scores,
          buffers.scratch_value, TILE_SIZE, shape);
      break;
  }
}

const char *kernelName(Kernel kernel) {
  switch (kernel) {
    case Kernel::Materialized:
      return "materialized_ref";
    case Kernel::TiledExact:
      return "tiled_online_exact";
    case Kernel::TiledFastExp:
      return "tiled_online_fast_exp";
    case Kernel::Int8Materialized:
      return "int8_materialized";
    case Kernel::Int8TiledExact:
      return "int8_tiled_online";
    case Kernel::Int8QKInt16VMaterialized:
      return "int8_qk_int16_v_materialized";
    case Kernel::Int8QKInt16VTiledExact:
      return "int8_qk_int16_v_tiled_online";
  }
  return "unknown";
}

uint8_t repetitionsFor(const AttentionShape &shape) {
  const uint32_t dot_products =
      static_cast<uint32_t>(shape.sequence) * shape.sequence *
      shape.head_dimension;
  if (dot_products >= 262144u) {
    return 3;
  }
  if (dot_products >= 65536u) {
    return 5;
  }
  return MAX_TIMING_SAMPLES;
}

uint64_t median(uint64_t *values, uint8_t count) {
  for (uint8_t index = 1; index < count; ++index) {
    const uint64_t value = values[index];
    uint8_t position = index;
    while (position > 0 && values[position - 1] > value) {
      values[position] = values[position - 1];
      --position;
    }
    values[position] = value;
  }
  return values[count / 2];
}

struct TimingStats {
  uint64_t median_us;
  uint64_t minimum_us;
};

TimingStats benchmarkKernel(Kernel kernel,
                            const AttentionShape &shape,
                            Buffers &buffers,
                            float *output) {
  runKernel(kernel, shape, buffers, output);
  const uint8_t repetitions = repetitionsFor(shape);
  uint64_t samples[MAX_TIMING_SAMPLES];
  uint64_t minimum = UINT64_MAX;

  for (uint8_t repetition = 0; repetition < repetitions; ++repetition) {
    const int64_t start = esp_timer_get_time();
    runKernel(kernel, shape, buffers, output);
    const uint64_t elapsed =
        static_cast<uint64_t>(esp_timer_get_time() - start);
    samples[repetition] = elapsed;
    if (elapsed < minimum) {
      minimum = elapsed;
    }
    yield();
  }

  return {median(samples, repetitions), minimum};
}

size_t kernelWorkingSetBytes(Kernel kernel, const AttentionShape &shape) {
  if (kernel == Kernel::Int8Materialized) {
    return materializedInt8WorkingSetBytes(shape);
  }
  if (kernel == Kernel::Int8TiledExact) {
    return tiledOnlineInt8WorkingSetBytes(shape, TILE_SIZE);
  }
  if (kernel == Kernel::Int8QKInt16VMaterialized) {
    return materializedInt8QKInt16VWorkingSetBytes(shape);
  }
  if (kernel == Kernel::Int8QKInt16VTiledExact) {
    return tiledOnlineInt8QKInt16VWorkingSetBytes(shape, TILE_SIZE);
  }
  const size_t activations =
      static_cast<size_t>(4) * shape.sequence * shape.head_dimension *
      sizeof(float);  // Q, K, V, output
  return activations +
         (kernel == Kernel::Materialized
              ? materializedWorkspaceBytes(shape)
              : tiledOnlineWorkspaceBytes(shape, TILE_SIZE));
}

size_t kernelWorkspaceBytes(Kernel kernel, const AttentionShape &shape) {
  return kernel == Kernel::Materialized ||
                 kernel == Kernel::Int8Materialized ||
                 kernel == Kernel::Int8QKInt16VMaterialized
             ? materializedWorkspaceBytes(shape)
             : tiledOnlineWorkspaceBytes(shape, TILE_SIZE);
}

void printResult(const AttentionShape &shape,
                 Kernel kernel,
                 const TimingStats &timing,
                 const AccuracyStats &accuracy) {
  const bool passed = accuracy.failed_elements == 0;
  Serial.printf(
      "RESULT,%u,%u,%u,%s,%llu,%llu,%u,%u,%.9g,%.9g,%lu,%s\n",
      shape.sequence, shape.head_dimension, shape.causal ? 1 : 0,
      kernelName(kernel),
      static_cast<unsigned long long>(timing.median_us),
      static_cast<unsigned long long>(timing.minimum_us),
      static_cast<unsigned int>(kernelWorkspaceBytes(kernel, shape)),
      static_cast<unsigned int>(kernelWorkingSetBytes(kernel, shape)),
      accuracy.max_absolute_error, accuracy.max_relative_error,
      static_cast<unsigned long>(accuracy.failed_elements),
      passed ? "PASS" : "FAIL");
}

void benchmarkShape(const AttentionShape &shape) {
  Buffers buffers;
  const size_t elements =
      static_cast<size_t>(shape.sequence) * shape.head_dimension;
  if (!allocateBuffers(shape, buffers)) {
    Serial.printf("SKIP,%u,%u,%u,allocation_failed,%u\n", shape.sequence,
                  shape.head_dimension, shape.causal ? 1 : 0,
                  static_cast<unsigned int>(ESP.getFreeHeap()));
    return;
  }

  fillInputs(buffers, elements);
  quantizeInputs(buffers, elements);
  runKernel(Kernel::Materialized, shape, buffers, buffers.reference);

  const AccuracyStats reference_accuracy =
      {0.0f, 0.0f, 0, static_cast<uint32_t>(elements)};
  const TimingStats reference_timing = benchmarkKernel(
      Kernel::Materialized, shape, buffers, buffers.reference);
  printResult(shape, Kernel::Materialized, reference_timing,
              reference_accuracy);

  const Kernel candidates[] = {
      Kernel::TiledExact,
      Kernel::TiledFastExp,
      Kernel::Int8Materialized,
      Kernel::Int8TiledExact,
      Kernel::Int8QKInt16VMaterialized,
      Kernel::Int8QKInt16VTiledExact,
  };
  for (Kernel kernel : candidates) {
    runKernel(kernel, shape, buffers, buffers.candidate);
    const AccuracyStats accuracy = compareAttentionOutputs(
        buffers.reference, buffers.candidate, elements, RELATIVE_TOLERANCE,
        ABSOLUTE_TOLERANCE);
    const TimingStats timing =
        benchmarkKernel(kernel, shape, buffers, buffers.candidate);
    printResult(shape, kernel, timing, accuracy);
  }

  releaseBuffers(buffers);
  Serial.printf("HEAP_AFTER,%u,%u,%u\n", shape.sequence,
                static_cast<unsigned int>(ESP.getFreeHeap()),
                static_cast<unsigned int>(ESP.getMinFreeHeap()));
}

void runBenchmarkSuite() {
  Serial.println("TECHJAM_ATTENTION_BENCHMARK_V3");
  Serial.printf("DEVICE,ESP32-C3,%uMHz,free_heap=%u,min_free_heap=%u\n",
                static_cast<unsigned int>(getCpuFrequencyMhz()),
                static_cast<unsigned int>(ESP.getFreeHeap()),
                static_cast<unsigned int>(ESP.getMinFreeHeap()));
  Serial.printf("CONFIG,tile=%u,rtol=%.6g,atol=%.6g,float_bytes=%u\n",
                TILE_SIZE, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE,
                static_cast<unsigned int>(sizeof(float)));
  Serial.println(
      "CSV,sequence,head_dimension,causal,kernel,median_us,min_us,"
      "workspace_bytes,working_set_bytes,max_abs_error,max_relative_error,"
      "failed_elements,status");

  for (const AttentionShape &shape : SHAPES) {
    benchmarkShape(shape);
  }

  Serial.printf("DONE,free_heap=%u,min_free_heap=%u\n",
                static_cast<unsigned int>(ESP.getFreeHeap()),
                static_cast<unsigned int>(ESP.getMinFreeHeap()));
  Serial.println("Send r to run the suite again.");
}

uint64_t benchmarkEndToEnd(EndToEndBuffers &buffers,
                           bool causal,
                           uint8_t candidate) {
  constexpr uint8_t REPETITIONS = 5;
  uint64_t samples[REPETITIONS];
  for (uint8_t repetition = 0; repetition < REPETITIONS; ++repetition) {
    const int64_t start = esp_timer_get_time();
    if (candidate == 1) {
      runEndToEndMixedTiled(END_TO_END_CONFIG, buffers, causal);
    } else if (candidate == 2) {
      runEndToEndIntProjectionMixedTiled(END_TO_END_CONFIG, buffers, causal);
    } else {
      runEndToEndFloatReference(END_TO_END_CONFIG, buffers, causal);
    }
    samples[repetition] =
        static_cast<uint64_t>(esp_timer_get_time() - start);
    yield();
  }
  return median(samples, REPETITIONS);
}

const char *endToEndCandidateName(uint8_t candidate) {
  return candidate == 2 ? "int16_act_int8_proj_mixed_attention"
                        : "float_proj_mixed_attention";
}

void runEndToEndCandidate(EndToEndBuffers &buffers,
                          bool causal,
                          uint8_t candidate) {
  if (candidate == 2) {
    runEndToEndIntProjectionMixedTiled(END_TO_END_CONFIG, buffers, causal);
  } else {
    runEndToEndMixedTiled(END_TO_END_CONFIG, buffers, causal);
  }
}

void emitEndToEndOutput(const EndToEndBuffers &buffers,
                        bool causal,
                        uint8_t candidate) {
  const size_t elements = static_cast<size_t>(END_TO_END_CONFIG.sequence) *
                          END_TO_END_CONFIG.model_dimension;
  for (size_t index = 0; index < elements; ++index) {
    Serial.printf("E2E_OUTPUT,%s,%u,%u,%.9g\n",
                  endToEndCandidateName(candidate), causal ? 1 : 0,
                  static_cast<unsigned int>(index), buffers.candidate[index]);
  }
}

void runEndToEndSuite() {
  Serial.println("TECHJAM_END_TO_END_ATTENTION_V2");
  Serial.printf(
      "E2E_CONFIG,sequence=%u,model_dimension=%u,heads=%u,head_dimension=%u,"
      "tile=%u,padding_rule=token_mod_7_not_5\n",
      END_TO_END_CONFIG.sequence, END_TO_END_CONFIG.model_dimension,
      END_TO_END_CONFIG.heads,
      END_TO_END_CONFIG.model_dimension / END_TO_END_CONFIG.heads,
      END_TO_END_CONFIG.tile_size);

  EndToEndBuffers buffers;
  if (!allocateEndToEndBuffers(END_TO_END_CONFIG, buffers)) {
    Serial.printf("E2E_ERROR,allocation_failed,free_heap=%u\n",
                  static_cast<unsigned int>(ESP.getFreeHeap()));
    Serial.println("E2E_DONE");
    return;
  }
  initializeEndToEndFixture(END_TO_END_CONFIG, buffers);
  const size_t elements = static_cast<size_t>(END_TO_END_CONFIG.sequence) *
                          END_TO_END_CONFIG.model_dimension;

  for (uint8_t causal_value = 0; causal_value <= 1; ++causal_value) {
    const bool causal = causal_value != 0;
    runEndToEndFloatReference(END_TO_END_CONFIG, buffers, causal);
    const uint64_t reference_us = benchmarkEndToEnd(buffers, causal, 0);
    for (uint8_t candidate = 1; candidate <= 2; ++candidate) {
      runEndToEndCandidate(buffers, causal, candidate);
      const AccuracyStats accuracy = compareAttentionOutputs(
          buffers.reference, buffers.candidate, elements, RELATIVE_TOLERANCE,
          ABSOLUTE_TOLERANCE);
      const uint64_t candidate_us =
          benchmarkEndToEnd(buffers, causal, candidate);
      // Re-run after timing so emitted values belong to this candidate.
      runEndToEndCandidate(buffers, causal, candidate);
      const size_t working_set =
          candidate == 2
              ? endToEndIntProjectionWorkingSetBytes(END_TO_END_CONFIG)
              : endToEndMixedWorkingSetBytes(END_TO_END_CONFIG);
      Serial.printf(
          "E2E_RESULT,%s,%u,%llu,%llu,%.6f,%u,%u,%.9g,%.9g,%lu,%s\n",
          endToEndCandidateName(candidate), causal ? 1 : 0,
          static_cast<unsigned long long>(reference_us),
          static_cast<unsigned long long>(candidate_us),
          static_cast<double>(reference_us) /
              static_cast<double>(candidate_us),
          static_cast<unsigned int>(
              endToEndMixedWorkspaceBytes(END_TO_END_CONFIG)),
          static_cast<unsigned int>(working_set),
          accuracy.max_absolute_error, accuracy.max_relative_error,
          static_cast<unsigned long>(accuracy.failed_elements),
          accuracy.failed_elements == 0 ? "PASS" : "FAIL");
      emitEndToEndOutput(buffers, causal, candidate);
    }
  }

  releaseEndToEndBuffers(buffers);
  Serial.printf("E2E_DONE,free_heap=%u,min_free_heap=%u\n",
                static_cast<unsigned int>(ESP.getFreeHeap()),
                static_cast<unsigned int>(ESP.getMinFreeHeap()));
  Serial.println("Send e to run the end-to-end suite again.");
}

}  // namespace

void setup() {
  Serial.begin(115200);
  const unsigned long serial_deadline = millis() + 5000;
  while (!Serial && static_cast<long>(serial_deadline - millis()) > 0) {
    delay(10);
  }
  delay(250);
  Serial.println("TECHJAM_ATTENTION_READY_V5");
  Serial.println("Send r for kernels or e for end-to-end attention.");
}

void loop() {
  if (Serial.available()) {
    const char command = static_cast<char>(Serial.read());
    if (command == 'r' || command == 'R') {
      runBenchmarkSuite();
    } else if (command == 'e' || command == 'E') {
      runEndToEndSuite();
    }
  }
  delay(20);
}
