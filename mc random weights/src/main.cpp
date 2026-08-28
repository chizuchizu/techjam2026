#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// Exact dimensions from torch_transformer_benchmark.py.
constexpr int BATCH = 8;
constexpr int SEQ = 128;
constexpr int DMODEL = 512;
constexpr int HEADS = 8;
constexpr int HEAD_DIM = DMODEL / HEADS;
constexpr int FFN = 2048;
constexpr int LAYERS = 6;
constexpr uint32_t SEED = 1234U;
constexpr bool CAUSAL = false;
constexpr float LAYER_NORM_EPS = 1.0e-5f;

// Fixed symmetric activation quantization. Real value = int8 value * scale.
constexpr float X_SCALE = 1.0f / 16.0f;
constexpr float NORM_SCALE = 1.0f / 32.0f;
constexpr float PROJECTION_SCALE = 1.0f / 32.0f;
constexpr float CONTEXT_SCALE = 1.0f / 64.0f;
constexpr float HIDDEN_SCALE = 1.0f / 32.0f;

constexpr int OUTPUT_TILE = 8;
constexpr int V_TOKEN_TILE = 16;
constexpr int FFN_TOKEN_TILE = 32;
constexpr size_t MATRIX_VALUES = (size_t)SEQ * DMODEL;

// Four 64 KiB matrices are the core working set (256 KiB total).
// They are allocated as one block at runtime because the C3 linker reserves less
// static DRAM than the runtime heap can expose. x is the residual stream; a/b/c
// are reused for norm, Q/K/V, context, and FFN hidden.
static int8_t *matrixMemory = nullptr;
static int8_t *xBuf = nullptr;
static int8_t *aBuf = nullptr;
static int8_t *bBuf = nullptr;
static int8_t *cBuf = nullptr;

// Small tiled accumulators and normalization scratch keep total SRAM under 320 KiB.
static int32_t accum[SEQ * OUTPUT_TILE];
static int8_t normTile[V_TOKEN_TILE * DMODEL];
static float scores[SEQ];

static uint64_t saturationCount = 0;
static uint64_t quantizedValueCount = 0;

static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static inline float uniform01(uint32_t key) {
  return (float)((mix32(key) >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static inline uint32_t parameterKey(int layer, int parameter, uint32_t index) {
  uint32_t key = SEED ^ 0x9e3779b9U;
  key ^= (uint32_t)(layer + 1) * 0x85ebca6bU;
  key ^= (uint32_t)(parameter + 1) * 0xc2b2ae35U;
  key ^= index * 0x27d4eb2dU;
  return key;
}

// INT8 weight values are generated directly from (seed, layer, tensor, index).
// No model weights are stored in SRAM, flash, or on the host.
static inline int8_t generatedWeightQ(int layer, int parameter, uint32_t index) {
  int value = (int)(mix32(parameterKey(layer, parameter, index)) >> 24) - 128;
  if (value < -127) value = -127;
  return (int8_t)value;
}

static inline float generatedBias(int layer, int parameter, uint32_t index,
                                  int fanIn) {
  const float bound = 1.0f / sqrtf((float)fanIn);
  return (2.0f * uniform01(parameterKey(layer, parameter + 1, index)) - 1.0f) *
         bound;
}

static inline float generatedNormal(uint32_t index) {
  const float u1 = uniform01(SEED ^ (index * 2U + 0x68bc21ebU));
  const float u2 = uniform01(SEED ^ (index * 2U + 0x02e5be93U));
  return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static inline int8_t quantize(float value, float scale) {
  int quantized = (int)lrintf(value / scale);
  ++quantizedValueCount;
  if (quantized > 127) {
    quantized = 127;
    ++saturationCount;
  } else if (quantized < -127) {
    quantized = -127;
    ++saturationCount;
  }
  return (int8_t)quantized;
}

static inline float geluExact(float value) {
  return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
}

static void generateInput(int batch) {
  const uint32_t base = (uint32_t)batch * (uint32_t)MATRIX_VALUES;
  for (uint32_t i = 0; i < MATRIX_VALUES; ++i)
    xBuf[i] = quantize(generatedNormal(base + i), X_SCALE);
}

static void layerNormRows(const int8_t *input, int8_t *output, int tokens,
                          float inputScale) {
  for (int token = 0; token < tokens; ++token) {
    const int8_t *in = input + (size_t)token * DMODEL;
    int8_t *out = output + (size_t)token * DMODEL;
    float meanQ = 0.0f;
    for (int d = 0; d < DMODEL; ++d) meanQ += (float)in[d];
    meanQ /= (float)DMODEL;

    float varianceQ = 0.0f;
    for (int d = 0; d < DMODEL; ++d) {
      const float delta = (float)in[d] - meanQ;
      varianceQ += delta * delta;
    }
    const float variance = varianceQ * inputScale * inputScale / (float)DMODEL;
    const float invStd = 1.0f / sqrtf(variance + LAYER_NORM_EPS);
    for (int d = 0; d < DMODEL; ++d) {
      const float normalized = ((float)in[d] - meanQ) * inputScale * invStd;
      out[d] = quantize(normalized, NORM_SCALE);
    }
  }
}

// Tiled int8 GEMM with int32 accumulation. Weights are generated once per call,
// output tile, and input feature, then reused across every token in the call.
static void linearQuantized(const int8_t *input, int8_t *output, int tokens,
                            int inFeatures, int outFeatures, int layer,
                            int parameter, float inputScale, float outputScale,
                            bool applyGelu, const int8_t *residual = nullptr,
                            float residualScale = 1.0f) {
  const float weightScale = (1.0f / sqrtf((float)inFeatures)) / 127.0f;

  for (int firstOut = 0; firstOut < outFeatures; firstOut += OUTPUT_TILE) {
    const int width = min(OUTPUT_TILE, outFeatures - firstOut);
    memset(accum, 0, (size_t)tokens * width * sizeof(int32_t));

    for (int i = 0; i < inFeatures; ++i) {
      int8_t weights[OUTPUT_TILE];
      for (int j = 0; j < width; ++j) {
        const int outputFeature = firstOut + j;
        const uint32_t index = (uint32_t)outputFeature * inFeatures + i;
        weights[j] = generatedWeightQ(layer, parameter, index);
      }
      for (int token = 0; token < tokens; ++token) {
        const int inputValue = input[(size_t)token * inFeatures + i];
        int32_t *rowAccum = accum + (size_t)token * width;
        for (int j = 0; j < width; ++j)
          rowAccum[j] += inputValue * (int)weights[j];
      }
    }

    for (int token = 0; token < tokens; ++token) {
      for (int j = 0; j < width; ++j) {
        const int outputFeature = firstOut + j;
        float value = (float)accum[(size_t)token * width + j] * inputScale *
                          weightScale +
                      generatedBias(layer, parameter, (uint32_t)outputFeature,
                                    inFeatures);
        if (applyGelu) value = geluExact(value);
        const size_t outputIndex = (size_t)token * outFeatures + outputFeature;
        if (residual != nullptr)
          value += (float)residual[outputIndex] * residualScale;
        output[outputIndex] = quantize(value, outputScale);
      }
    }
    yield();
  }
}

static void buildValueProjection(int layer) {
  // aBuf currently holds normalized x. Q and K are already in bBuf and cBuf.
  // Re-normalize small x tiles so V can overwrite aBuf without a fifth matrix.
  for (int first = 0; first < SEQ; first += V_TOKEN_TILE) {
    const int tokens = min(V_TOKEN_TILE, SEQ - first);
    layerNormRows(xBuf + (size_t)first * DMODEL, normTile, tokens, X_SCALE);
    linearQuantized(normTile, aBuf + (size_t)first * DMODEL, tokens, DMODEL,
                    DMODEL, layer, 4, NORM_SCALE, PROJECTION_SCALE, false);
  }
}

static void attentionInPlace() {
  // bBuf=Q, cBuf=K, aBuf=V. Each completed Q row is replaced by context.
  const float scoreScale =
      PROJECTION_SCALE * PROJECTION_SCALE / sqrtf((float)HEAD_DIM);
  for (int query = 0; query < SEQ; ++query) {
    for (int head = 0; head < HEADS; ++head) {
      const int offset = head * HEAD_DIM;
      float maximum = -INFINITY;
      for (int key = 0; key < SEQ; ++key) {
        if (CAUSAL && key > query) {
          scores[key] = -INFINITY;
          continue;
        }
        int32_t dot = 0;
        for (int d = 0; d < HEAD_DIM; ++d)
          dot += (int)bBuf[(size_t)query * DMODEL + offset + d] *
                 (int)cBuf[(size_t)key * DMODEL + offset + d];
        scores[key] = (float)dot * scoreScale;
        if (scores[key] > maximum) maximum = scores[key];
      }

      float denominator = 0.0f;
      for (int key = 0; key < SEQ; ++key) {
        if (isinf(scores[key]) && scores[key] < 0.0f) {
          scores[key] = 0.0f;
        } else {
          scores[key] = expf(scores[key] - maximum);
          denominator += scores[key];
        }
      }

      for (int d = 0; d < HEAD_DIM; ++d) {
        float context = 0.0f;
        for (int key = 0; key < SEQ; ++key)
          context += (scores[key] / denominator) *
                     ((float)aBuf[(size_t)key * DMODEL + offset + d] *
                      PROJECTION_SCALE);
        bBuf[(size_t)query * DMODEL + offset + d] =
            quantize(context, CONTEXT_SCALE);
      }
    }
    if ((query & 15) == 15) yield();
  }
}

static void feedForward(int layer) {
  // Attention no longer needs Q/K/V: bBuf becomes normalized residual input and
  // aBuf is reused as a 32x2048 hidden tile (exactly 64 KiB).
  layerNormRows(xBuf, bBuf, SEQ, X_SCALE);
  for (int first = 0; first < SEQ; first += FFN_TOKEN_TILE) {
    const int tokens = min(FFN_TOKEN_TILE, SEQ - first);
    linearQuantized(bBuf + (size_t)first * DMODEL, aBuf, tokens, DMODEL, FFN,
                    layer, 8, NORM_SCALE, HIDDEN_SCALE, true);
    linearQuantized(aBuf, xBuf + (size_t)first * DMODEL, tokens, FFN, DMODEL,
                    layer, 10, HIDDEN_SCALE, X_SCALE, false,
                    xBuf + (size_t)first * DMODEL, X_SCALE);
  }
}

static void transformerLayer(int layer) {
  layerNormRows(xBuf, aBuf, SEQ, X_SCALE);
  linearQuantized(aBuf, bBuf, SEQ, DMODEL, DMODEL, layer, 0, NORM_SCALE,
                  PROJECTION_SCALE, false);  // Q
  linearQuantized(aBuf, cBuf, SEQ, DMODEL, DMODEL, layer, 2, NORM_SCALE,
                  PROJECTION_SCALE, false);  // K
  buildValueProjection(layer);              // V in aBuf
  attentionInPlace();                        // context replaces Q in bBuf
  linearQuantized(bBuf, xBuf, SEQ, DMODEL, DMODEL, layer, 6, CONTEXT_SCALE,
                  X_SCALE, false, xBuf, X_SCALE);
  feedForward(layer);
}

static double finalChecksum() {
  layerNormRows(xBuf, bBuf, SEQ, X_SCALE);
  double checksum = 0.0;
  for (size_t i = 0; i < MATRIX_VALUES; ++i)
    checksum += (double)bBuf[i] * (double)NORM_SCALE *
                (double)((i % 17U) + 1U);
  return checksum;
}

static uint64_t logicalParameterCount() {
  const uint64_t perLayer =
      4ULL * DMODEL * DMODEL + 4ULL * DMODEL +
      2ULL * DMODEL * FFN + FFN + DMODEL + 4ULL * DMODEL;
  return perLayer * LAYERS + 2ULL * DMODEL;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== On-device INT8 Transformer ===");
  const size_t matrixBytes = 4 * MATRIX_VALUES * sizeof(int8_t);
  Serial.printf("free heap before matrix allocation=%u bytes\n", ESP.getFreeHeap());
  matrixMemory = static_cast<int8_t *>(malloc(matrixBytes));
  if (matrixMemory == nullptr) {
    Serial.printf("ERROR: unable to allocate %u-byte matrix workspace\n",
                  (unsigned)matrixBytes);
    return;
  }
  xBuf = matrixMemory;
  aBuf = xBuf + MATRIX_VALUES;
  bBuf = aBuf + MATRIX_VALUES;
  cBuf = bBuf + MATRIX_VALUES;

  Serial.printf("B=%d S=%d D=%d H=%d F=%d L=%d seed=%lu causal=%s\n", BATCH,
                SEQ, DMODEL, HEADS, FFN, LAYERS, (unsigned long)SEED,
                CAUSAL ? "true" : "false");
  Serial.printf("logical parameters=%llu; stored weight bytes=0\n",
                (unsigned long long)logicalParameterCount());
  Serial.printf("quantized matrix buffers=%u bytes; free heap=%u bytes\n",
                (unsigned)(4 * MATRIX_VALUES), ESP.getFreeHeap());
  Serial.println("INT8 activations/weights, INT32 accumulators, float LN/softmax/GELU");

  const uint32_t fullStart = millis();
  double checksum = 0.0;
  for (int batch = 0; batch < BATCH; ++batch) {
    Serial.printf("batch %d/%d: generating input\n", batch + 1, BATCH);
    generateInput(batch);
    for (int layer = 0; layer < LAYERS; ++layer) {
      const uint32_t layerStart = millis();
      Serial.printf("  layer %d/%d...\n", layer + 1, LAYERS);
      transformerLayer(layer);
      Serial.printf("  layer %d complete: %.3f s\n", layer + 1,
                    (millis() - layerStart) / 1000.0f);
    }
    const double batchChecksum = finalChecksum();
    checksum += batchChecksum;
    Serial.printf("batch %d checksum=%.9g elapsed=%.1f s\n", batch + 1,
                  batchChecksum, (millis() - fullStart) / 1000.0f);
    yield();
  }

  const double saturationPercent = quantizedValueCount == 0
                                       ? 0.0
                                       : 100.0 * (double)saturationCount /
                                             (double)quantizedValueCount;
  const float elapsedSeconds = (millis() - fullStart) / 1000.0f;
  Serial.println("\n=== Result ===");
  Serial.printf("checksum=%.9g\n", checksum);
  Serial.printf("latency=%.3f s throughput=%.3f token/s\n", elapsedSeconds,
                (BATCH * SEQ) / elapsedSeconds);
  Serial.printf("saturation=%llu/%llu (%.5f%%)\n",
                (unsigned long long)saturationCount,
                (unsigned long long)quantizedValueCount, saturationPercent);
  Serial.printf("free heap=%u bytes\n", ESP.getFreeHeap());
}

void loop() { delay(1000); }
