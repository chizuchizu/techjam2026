#include "tiny_transformer.h"

#include <math.h>
#include <stdlib.h>

namespace {

using namespace tiny_transformer_weights;

constexpr size_t MATRIX_ELEMENTS =
    static_cast<size_t>(CONTEXT) * MODEL_DIMENSION;
constexpr size_t HIDDEN_ELEMENTS =
    static_cast<size_t>(CONTEXT) * FFN_DIMENSION;

struct LayerWeights {
  const float *norm1;
  const float *norm2;
  const int8_t *query_weight;
  const float *query_scale;
  const int8_t *key_weight;
  const float *key_scale;
  const int8_t *value_weight;
  const float *value_scale;
  const int8_t *output_weight;
  const float *output_scale;
  const int8_t *ffn1_weight;
  const float *ffn1_scale;
  const int8_t *ffn2_weight;
  const float *ffn2_scale;
};

const LayerWeights LAYER_WEIGHTS[LAYERS] = {
    {LAYER0_NORM1, LAYER0_NORM2, LAYER0_QUERY_WEIGHT, LAYER0_QUERY_SCALE,
     LAYER0_KEY_WEIGHT, LAYER0_KEY_SCALE, LAYER0_VALUE_WEIGHT,
     LAYER0_VALUE_SCALE, LAYER0_OUTPUT_WEIGHT, LAYER0_OUTPUT_SCALE,
     LAYER0_FFN1_WEIGHT, LAYER0_FFN1_SCALE, LAYER0_FFN2_WEIGHT,
     LAYER0_FFN2_SCALE},
    {LAYER1_NORM1, LAYER1_NORM2, LAYER1_QUERY_WEIGHT, LAYER1_QUERY_SCALE,
     LAYER1_KEY_WEIGHT, LAYER1_KEY_SCALE, LAYER1_VALUE_WEIGHT,
     LAYER1_VALUE_SCALE, LAYER1_OUTPUT_WEIGHT, LAYER1_OUTPUT_SCALE,
     LAYER1_FFN1_WEIGHT, LAYER1_FFN1_SCALE, LAYER1_FFN2_WEIGHT,
     LAYER1_FFN2_SCALE},
};

template <typename T>
bool allocate(T *&pointer, size_t elements) {
  pointer = static_cast<T *>(malloc(elements * sizeof(T)));
  return pointer != nullptr;
}

void rmsNorm(const float *source,
             const float *weight,
             float *destination,
             uint16_t rows) {
  for (uint16_t row = 0; row < rows; ++row) {
    const size_t offset = static_cast<size_t>(row) * MODEL_DIMENSION;
    float sum_square = 0.0f;
    for (uint16_t column = 0; column < MODEL_DIMENSION; ++column) {
      const float value = source[offset + column];
      sum_square += value * value;
    }
    const float inverse_rms =
        1.0f / sqrtf(sum_square / MODEL_DIMENSION + 1.0e-5f);
    for (uint16_t column = 0; column < MODEL_DIMENSION; ++column) {
      destination[offset + column] =
          source[offset + column] * inverse_rms * weight[column];
    }
  }
}

float quantizeInt16(const float *source,
                    int16_t *destination,
                    size_t elements) {
  float maximum = 0.0f;
  for (size_t index = 0; index < elements; ++index) {
    maximum = fmaxf(maximum, fabsf(source[index]));
  }
  const float scale = maximum > 0.0f ? maximum / 32767.0f : 1.0f;
  const float inverse_scale = 1.0f / scale;
  for (size_t index = 0; index < elements; ++index) {
    int32_t quantized = lroundf(source[index] * inverse_scale);
    if (quantized > 32767) quantized = 32767;
    if (quantized < -32767) quantized = -32767;
    destination[index] = static_cast<int16_t>(quantized);
  }
  return scale;
}

float quantizeInt8(const float *source,
                   int8_t *destination,
                   size_t elements) {
  float maximum = 0.0f;
  for (size_t index = 0; index < elements; ++index) {
    maximum = fmaxf(maximum, fabsf(source[index]));
  }
  const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
  const float inverse_scale = 1.0f / scale;
  for (size_t index = 0; index < elements; ++index) {
    int32_t quantized = lroundf(source[index] * inverse_scale);
    if (quantized > 127) quantized = 127;
    if (quantized < -127) quantized = -127;
    destination[index] = static_cast<int8_t>(quantized);
  }
  return scale;
}

void quantizedLinear(const float *source,
                     const int8_t *weight,
                     const float *weight_scale,
                     float *destination,
                     uint16_t rows,
                     uint16_t input_columns,
                     uint16_t output_columns,
                     int16_t *activation_int16) {
  const size_t source_elements = static_cast<size_t>(rows) * input_columns;
  const float activation_scale =
      quantizeInt16(source, activation_int16, source_elements);
  for (uint16_t row = 0; row < rows; ++row) {
    for (uint16_t output = 0; output < output_columns; ++output) {
      int32_t accumulator = 0;
      for (uint16_t input = 0; input < input_columns; ++input) {
        const size_t activation_index =
            static_cast<size_t>(row) * input_columns + input;
        const size_t weight_index =
            static_cast<size_t>(input) * output_columns + output;
        accumulator += static_cast<int32_t>(activation_int16[activation_index]) *
                       static_cast<int32_t>(weight[weight_index]);
      }
      destination[static_cast<size_t>(row) * output_columns + output] =
          static_cast<float>(accumulator) * activation_scale *
          weight_scale[output];
    }
  }
}

void mixedCausalAttention(TinyTransformerBuffers &buffers) {
  const float query_scale =
      quantizeInt8(buffers.query, buffers.query_int8, MATRIX_ELEMENTS);
  const float key_scale =
      quantizeInt8(buffers.key, buffers.key_int8, MATRIX_ELEMENTS);
  const float value_scale =
      quantizeInt16(buffers.value, buffers.value_int16, MATRIX_ELEMENTS);
  const float score_scale =
      query_scale * key_scale / sqrtf(static_cast<float>(HEAD_DIMENSION));

  for (uint16_t head = 0; head < HEADS; ++head) {
    const uint16_t head_offset = head * HEAD_DIMENSION;
    for (uint16_t query_index = 0; query_index < CONTEXT; ++query_index) {
      float maximum = -INFINITY;
      for (uint16_t key_index = 0; key_index <= query_index; ++key_index) {
        int32_t dot = 0;
        for (uint16_t column = 0; column < HEAD_DIMENSION; ++column) {
          const size_t query_offset =
              static_cast<size_t>(query_index) * MODEL_DIMENSION +
              head_offset + column;
          const size_t key_offset =
              static_cast<size_t>(key_index) * MODEL_DIMENSION +
              head_offset + column;
          dot += static_cast<int32_t>(buffers.query_int8[query_offset]) *
                 static_cast<int32_t>(buffers.key_int8[key_offset]);
        }
        const float score = static_cast<float>(dot) * score_scale;
        buffers.scores[key_index] = score;
        maximum = fmaxf(maximum, score);
      }

      float denominator = 0.0f;
      for (uint16_t key_index = 0; key_index <= query_index; ++key_index) {
        const float attention_weight =
            expf(buffers.scores[key_index] - maximum);
        buffers.scores[key_index] = attention_weight;
        denominator += attention_weight;
      }
      for (uint16_t column = 0; column < HEAD_DIMENSION; ++column) {
        float numerator = 0.0f;
        for (uint16_t key_index = 0; key_index <= query_index; ++key_index) {
          const size_t value_offset =
              static_cast<size_t>(key_index) * MODEL_DIMENSION +
              head_offset + column;
          numerator += buffers.scores[key_index] *
                       static_cast<float>(buffers.value_int16[value_offset]);
        }
        const size_t output_offset =
            static_cast<size_t>(query_index) * MODEL_DIMENSION +
            head_offset + column;
        buffers.context[output_offset] = numerator * value_scale / denominator;
      }
    }
  }
}

}  // namespace

bool allocateTinyTransformerBuffers(TinyTransformerBuffers &buffers) {
  const bool allocated =
      allocate(buffers.values, MATRIX_ELEMENTS) &&
      allocate(buffers.normalized, MATRIX_ELEMENTS) &&
      allocate(buffers.query, MATRIX_ELEMENTS) &&
      allocate(buffers.key, MATRIX_ELEMENTS) &&
      allocate(buffers.value, MATRIX_ELEMENTS) &&
      allocate(buffers.context, MATRIX_ELEMENTS) &&
      allocate(buffers.projection, MATRIX_ELEMENTS) &&
      allocate(buffers.hidden, HIDDEN_ELEMENTS) &&
      allocate(buffers.activation_int16, HIDDEN_ELEMENTS) &&
      allocate(buffers.query_int8, MATRIX_ELEMENTS) &&
      allocate(buffers.key_int8, MATRIX_ELEMENTS) &&
      allocate(buffers.value_int16, MATRIX_ELEMENTS) &&
      allocate(buffers.scores, CONTEXT);
  if (!allocated) releaseTinyTransformerBuffers(buffers);
  return allocated;
}

void releaseTinyTransformerBuffers(TinyTransformerBuffers &buffers) {
  free(buffers.values);
  free(buffers.normalized);
  free(buffers.query);
  free(buffers.key);
  free(buffers.value);
  free(buffers.context);
  free(buffers.projection);
  free(buffers.hidden);
  free(buffers.activation_int16);
  free(buffers.query_int8);
  free(buffers.key_int8);
  free(buffers.value_int16);
  free(buffers.scores);
  buffers = TinyTransformerBuffers{};
}

void runTinyTransformer(const uint8_t *tokens,
                        float *logits,
                        TinyTransformerBuffers &buffers) {
  for (uint16_t position = 0; position < CONTEXT; ++position) {
    const uint8_t token = tokens[position];
    for (uint16_t column = 0; column < MODEL_DIMENSION; ++column) {
      const size_t destination =
          static_cast<size_t>(position) * MODEL_DIMENSION + column;
      buffers.values[destination] =
          TOKEN_EMBEDDING[static_cast<size_t>(token) * MODEL_DIMENSION + column] +
          POSITION_EMBEDDING[destination];
    }
  }

  for (uint16_t layer = 0; layer < LAYERS; ++layer) {
    const LayerWeights &weights = LAYER_WEIGHTS[layer];
    rmsNorm(buffers.values, weights.norm1, buffers.normalized, CONTEXT);
    quantizedLinear(buffers.normalized, weights.query_weight,
                    weights.query_scale, buffers.query, CONTEXT,
                    MODEL_DIMENSION, MODEL_DIMENSION,
                    buffers.activation_int16);
    quantizedLinear(buffers.normalized, weights.key_weight, weights.key_scale,
                    buffers.key, CONTEXT, MODEL_DIMENSION, MODEL_DIMENSION,
                    buffers.activation_int16);
    quantizedLinear(buffers.normalized, weights.value_weight,
                    weights.value_scale, buffers.value, CONTEXT,
                    MODEL_DIMENSION, MODEL_DIMENSION,
                    buffers.activation_int16);
    mixedCausalAttention(buffers);
    quantizedLinear(buffers.context, weights.output_weight,
                    weights.output_scale, buffers.projection, CONTEXT,
                    MODEL_DIMENSION, MODEL_DIMENSION,
                    buffers.activation_int16);
    for (size_t index = 0; index < MATRIX_ELEMENTS; ++index) {
      buffers.values[index] += buffers.projection[index];
    }

    rmsNorm(buffers.values, weights.norm2, buffers.normalized, CONTEXT);
    quantizedLinear(buffers.normalized, weights.ffn1_weight,
                    weights.ffn1_scale, buffers.hidden, CONTEXT,
                    MODEL_DIMENSION, FFN_DIMENSION,
                    buffers.activation_int16);
    for (size_t index = 0; index < HIDDEN_ELEMENTS; ++index) {
      buffers.hidden[index] = fmaxf(buffers.hidden[index], 0.0f);
    }
    quantizedLinear(buffers.hidden, weights.ffn2_weight, weights.ffn2_scale,
                    buffers.projection, CONTEXT, FFN_DIMENSION,
                    MODEL_DIMENSION, buffers.activation_int16);
    for (size_t index = 0; index < MATRIX_ELEMENTS; ++index) {
      buffers.values[index] += buffers.projection[index];
    }
  }

  rmsNorm(buffers.values, FINAL_NORM, buffers.normalized, CONTEXT);
  const float *last =
      buffers.normalized + static_cast<size_t>(CONTEXT - 1) * MODEL_DIMENSION;
  quantizedLinear(last, LM_HEAD_WEIGHT, LM_HEAD_SCALE, logits, 1,
                  MODEL_DIMENSION, VOCABULARY, buffers.activation_int16);
}

size_t tinyTransformerWeightBytes() {
  size_t bytes = sizeof(TOKEN_EMBEDDING) + sizeof(POSITION_EMBEDDING) +
                 sizeof(FINAL_NORM) + sizeof(LM_HEAD_WEIGHT) +
                 sizeof(LM_HEAD_SCALE);
  for (uint16_t layer = 0; layer < LAYERS; ++layer) {
    const LayerWeights &weights = LAYER_WEIGHTS[layer];
    (void)weights;
    bytes += 2 * MODEL_DIMENSION * sizeof(float);
    bytes += 4 * MODEL_DIMENSION * MODEL_DIMENSION * sizeof(int8_t);
    bytes += 4 * MODEL_DIMENSION * sizeof(float);
    bytes += MODEL_DIMENSION * FFN_DIMENSION * sizeof(int8_t);
    bytes += FFN_DIMENSION * MODEL_DIMENSION * sizeof(int8_t);
    bytes += (FFN_DIMENSION + MODEL_DIMENSION) * sizeof(float);
  }
  return bytes;
}

size_t tinyTransformerWorkingSetBytes() {
  return 7 * MATRIX_ELEMENTS * sizeof(float) +
         HIDDEN_ELEMENTS * sizeof(float) +
         HIDDEN_ELEMENTS * sizeof(int16_t) + 2 * MATRIX_ELEMENTS * sizeof(int8_t) +
         MATRIX_ELEMENTS * sizeof(int16_t) + CONTEXT * sizeof(float) +
         VOCABULARY * sizeof(float);
}

uint8_t tinyTransformerArgmax(const float *logits) {
  uint8_t best = 0;
  for (uint8_t token = 1; token < VOCABULARY; ++token) {
    if (logits[token] > logits[best]) best = token;
  }
  return best;
}

int16_t tinyTransformerTokenForByte(uint8_t value) {
  for (uint8_t token = 0; token < VOCABULARY; ++token) {
    if (VOCAB_BYTES[token] == value) return token;
  }
  return -1;
}
