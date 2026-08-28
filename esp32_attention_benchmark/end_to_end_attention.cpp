#include "end_to_end_attention.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr uint8_t QUERY_PROJECTION = 0;
constexpr uint8_t KEY_PROJECTION = 1;
constexpr uint8_t VALUE_PROJECTION = 2;
constexpr uint8_t OUTPUT_PROJECTION = 3;

size_t activationElements(const EndToEndConfig &config) {
  return static_cast<size_t>(config.sequence) * config.model_dimension;
}

size_t matrixElements(const EndToEndConfig &config) {
  return static_cast<size_t>(config.model_dimension) * config.model_dimension;
}

float *matrix(EndToEndBuffers &buffers,
              const EndToEndConfig &config,
              uint8_t projection) {
  return buffers.weights + matrixElements(config) * projection;
}

float *bias(EndToEndBuffers &buffers,
            const EndToEndConfig &config,
            uint8_t projection) {
  return buffers.biases + static_cast<size_t>(config.model_dimension) *
                              projection;
}

int8_t *matrixInt8(EndToEndBuffers &buffers,
                   const EndToEndConfig &config,
                   uint8_t projection) {
  return buffers.weights_int8 + matrixElements(config) * projection;
}

float *weightScales(EndToEndBuffers &buffers,
                    const EndToEndConfig &config,
                    uint8_t projection) {
  return buffers.weight_scales +
         static_cast<size_t>(config.model_dimension) * projection;
}

void project(const float *input,
             const float *weights,
             const float *biases,
             float *output,
             const EndToEndConfig &config) {
  const uint16_t dimension = config.model_dimension;
  for (uint16_t token = 0; token < config.sequence; ++token) {
    const float *input_row = input + static_cast<size_t>(token) * dimension;
    float *output_row = output + static_cast<size_t>(token) * dimension;
    for (uint16_t output_feature = 0; output_feature < dimension;
         ++output_feature) {
      float sum = biases[output_feature];
      for (uint16_t input_feature = 0; input_feature < dimension;
           ++input_feature) {
        sum += input_row[input_feature] *
               weights[static_cast<size_t>(input_feature) * dimension +
                       output_feature];
      }
      output_row[output_feature] = sum;
    }
  }
}

float quantizeInt8(const float *source, int8_t *destination, size_t elements) {
  float maximum = 0.0f;
  for (size_t index = 0; index < elements; ++index) {
    maximum = fmaxf(maximum, fabsf(source[index]));
  }
  const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
  const float inverse_scale = 1.0f / scale;
  for (size_t index = 0; index < elements; ++index) {
    int32_t value = static_cast<int32_t>(lroundf(source[index] * inverse_scale));
    value = value > 127 ? 127 : value;
    value = value < -127 ? -127 : value;
    destination[index] = static_cast<int8_t>(value);
  }
  return scale;
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
    int32_t value = static_cast<int32_t>(lroundf(source[index] * inverse_scale));
    value = value > 32767 ? 32767 : value;
    value = value < -32767 ? -32767 : value;
    destination[index] = static_cast<int16_t>(value);
  }
  return scale;
}

void quantizeWeightColumns(const float *source,
                           int8_t *destination,
                           float *scales,
                           uint16_t dimension) {
  for (uint16_t output_feature = 0; output_feature < dimension;
       ++output_feature) {
    float maximum = 0.0f;
    for (uint16_t input_feature = 0; input_feature < dimension;
         ++input_feature) {
      maximum = fmaxf(
          maximum,
          fabsf(source[static_cast<size_t>(input_feature) * dimension +
                       output_feature]));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    scales[output_feature] = scale;
    const float inverse_scale = 1.0f / scale;
    for (uint16_t input_feature = 0; input_feature < dimension;
         ++input_feature) {
      const size_t index =
          static_cast<size_t>(input_feature) * dimension + output_feature;
      int32_t value =
          static_cast<int32_t>(lroundf(source[index] * inverse_scale));
      value = value > 127 ? 127 : value;
      value = value < -127 ? -127 : value;
      destination[index] = static_cast<int8_t>(value);
    }
  }
}

void projectInt16Int8(const int16_t *input,
                      float input_scale,
                      const int8_t *weights,
                      const float *weight_scales,
                      const float *biases,
                      float *output,
                      const EndToEndConfig &config) {
  const uint16_t dimension = config.model_dimension;
  for (uint16_t token = 0; token < config.sequence; ++token) {
    const int16_t *input_row =
        input + static_cast<size_t>(token) * dimension;
    float *output_row = output + static_cast<size_t>(token) * dimension;
    for (uint16_t output_feature = 0; output_feature < dimension;
         ++output_feature) {
      int32_t dot = 0;
      for (uint16_t input_feature = 0; input_feature < dimension;
           ++input_feature) {
        dot += static_cast<int32_t>(input_row[input_feature]) *
               weights[static_cast<size_t>(input_feature) * dimension +
                       output_feature];
      }
      output_row[output_feature] =
          biases[output_feature] +
          static_cast<float>(dot) * input_scale *
              weight_scales[output_feature];
    }
  }
}

inline bool keyIsVisible(uint16_t query, uint16_t key, bool causal) {
  return endToEndTokenIsValid(key) && (!causal || key <= query);
}

void attentionFloat(const EndToEndConfig &config,
                    EndToEndBuffers &buffers,
                    bool causal) {
  const uint16_t dimension = config.model_dimension;
  const uint16_t head_dimension = dimension / config.heads;
  const float score_scale = 1.0f / sqrtf(static_cast<float>(head_dimension));
  memset(buffers.context, 0, activationElements(config) * sizeof(float));

  for (uint16_t head = 0; head < config.heads; ++head) {
    const uint16_t head_offset = head * head_dimension;
    for (uint16_t query_index = 0; query_index < config.sequence;
         ++query_index) {
      if (!endToEndTokenIsValid(query_index)) {
        continue;
      }
      const float *query_row =
          buffers.query + static_cast<size_t>(query_index) * dimension +
          head_offset;
      float maximum = -FLT_MAX;

      for (uint16_t key_index = 0; key_index < config.sequence; ++key_index) {
        if (!keyIsVisible(query_index, key_index, causal)) {
          buffers.reference_scores[key_index] = -FLT_MAX;
          continue;
        }
        const float *key_row =
            buffers.key + static_cast<size_t>(key_index) * dimension +
            head_offset;
        float score = 0.0f;
        for (uint16_t feature = 0; feature < head_dimension; ++feature) {
          score += query_row[feature] * key_row[feature];
        }
        score *= score_scale;
        buffers.reference_scores[key_index] = score;
        maximum = fmaxf(maximum, score);
      }

      float sum = 0.0f;
      for (uint16_t key_index = 0; key_index < config.sequence; ++key_index) {
        if (!keyIsVisible(query_index, key_index, causal)) {
          continue;
        }
        const float probability =
            expf(buffers.reference_scores[key_index] - maximum);
        buffers.reference_scores[key_index] = probability;
        sum += probability;
      }

      float *context_row =
          buffers.context + static_cast<size_t>(query_index) * dimension +
          head_offset;
      const float inverse_sum = 1.0f / sum;
      for (uint16_t key_index = 0; key_index < config.sequence; ++key_index) {
        if (!keyIsVisible(query_index, key_index, causal)) {
          continue;
        }
        const float probability =
            buffers.reference_scores[key_index] * inverse_sum;
        const float *value_row =
            buffers.value + static_cast<size_t>(key_index) * dimension +
            head_offset;
        for (uint16_t feature = 0; feature < head_dimension; ++feature) {
          context_row[feature] += probability * value_row[feature];
        }
      }
    }
  }
}

void attentionMixedTiled(const EndToEndConfig &config,
                         EndToEndBuffers &buffers,
                         bool causal,
                         float query_scale,
                         float key_scale,
                         float value_scale) {
  const uint16_t dimension = config.model_dimension;
  const uint16_t head_dimension = dimension / config.heads;
  const float score_scale = query_scale * key_scale /
                            sqrtf(static_cast<float>(head_dimension));
  memset(buffers.context, 0, activationElements(config) * sizeof(float));

  for (uint16_t head = 0; head < config.heads; ++head) {
    const uint16_t head_offset = head * head_dimension;
    for (uint16_t query_index = 0; query_index < config.sequence;
         ++query_index) {
      if (!endToEndTokenIsValid(query_index)) {
        continue;
      }
      const int8_t *query_row =
          buffers.query_int8 + static_cast<size_t>(query_index) * dimension +
          head_offset;
      float *context_row =
          buffers.context + static_cast<size_t>(query_index) * dimension +
          head_offset;
      float running_max = -FLT_MAX;
      float running_sum = 0.0f;
      bool has_values = false;

      for (uint16_t tile_begin = 0; tile_begin < config.sequence;
           tile_begin += config.tile_size) {
        const uint16_t tile_end =
            static_cast<uint16_t>(tile_begin + config.tile_size < config.sequence
                                      ? tile_begin + config.tile_size
                                      : config.sequence);
        uint16_t tile_count = 0;
        float tile_max = -FLT_MAX;
        for (uint16_t key_index = tile_begin; key_index < tile_end;
             ++key_index) {
          if (!keyIsVisible(query_index, key_index, causal)) {
            continue;
          }
          const int8_t *key_row =
              buffers.key_int8 + static_cast<size_t>(key_index) * dimension +
              head_offset;
          int32_t dot = 0;
          for (uint16_t feature = 0; feature < head_dimension; ++feature) {
            dot += static_cast<int32_t>(query_row[feature]) * key_row[feature];
          }
          const float score = static_cast<float>(dot) * score_scale;
          buffers.scratch_scores[tile_count * 2] = score;
          buffers.scratch_scores[tile_count * 2 + 1] =
              static_cast<float>(key_index);
          tile_max = fmaxf(tile_max, score);
          ++tile_count;
        }
        if (tile_count == 0) {
          continue;
        }

        memset(buffers.scratch_value, 0,
               static_cast<size_t>(head_dimension) * sizeof(float));
        float tile_sum = 0.0f;
        for (uint16_t offset = 0; offset < tile_count; ++offset) {
          const float weight =
              expf(buffers.scratch_scores[offset * 2] - tile_max);
          const uint16_t key_index =
              static_cast<uint16_t>(buffers.scratch_scores[offset * 2 + 1]);
          tile_sum += weight;
          const int16_t *value_row =
              buffers.value_int16 + static_cast<size_t>(key_index) * dimension +
              head_offset;
          for (uint16_t feature = 0; feature < head_dimension; ++feature) {
            buffers.scratch_value[feature] += weight * value_row[feature];
          }
        }

        if (!has_values) {
          memcpy(context_row, buffers.scratch_value,
                 static_cast<size_t>(head_dimension) * sizeof(float));
          running_max = tile_max;
          running_sum = tile_sum;
          has_values = true;
          continue;
        }

        const float merged_max = fmaxf(running_max, tile_max);
        const float old_scale = expf(running_max - merged_max);
        const float tile_scale = expf(tile_max - merged_max);
        for (uint16_t feature = 0; feature < head_dimension; ++feature) {
          context_row[feature] = context_row[feature] * old_scale +
                                 buffers.scratch_value[feature] * tile_scale;
        }
        running_sum = running_sum * old_scale + tile_sum * tile_scale;
        running_max = merged_max;
      }

      if (has_values) {
        const float output_scale = value_scale / running_sum;
        for (uint16_t feature = 0; feature < head_dimension; ++feature) {
          context_row[feature] *= output_scale;
        }
      }
    }
  }
}

void zeroPaddedRows(float *output, const EndToEndConfig &config) {
  for (uint16_t token = 0; token < config.sequence; ++token) {
    if (!endToEndTokenIsValid(token)) {
      memset(output + static_cast<size_t>(token) * config.model_dimension, 0,
             static_cast<size_t>(config.model_dimension) * sizeof(float));
    }
  }
}

void runProjections(const EndToEndConfig &config,
                    EndToEndBuffers &buffers) {
  project(buffers.input, matrix(buffers, config, QUERY_PROJECTION),
          bias(buffers, config, QUERY_PROJECTION), buffers.query, config);
  project(buffers.input, matrix(buffers, config, KEY_PROJECTION),
          bias(buffers, config, KEY_PROJECTION), buffers.key, config);
  project(buffers.input, matrix(buffers, config, VALUE_PROJECTION),
          bias(buffers, config, VALUE_PROJECTION), buffers.value, config);
}

void runIntProjections(const EndToEndConfig &config,
                       EndToEndBuffers &buffers,
                       float input_scale) {
  projectInt16Int8(
      buffers.input_int16, input_scale,
      matrixInt8(buffers, config, QUERY_PROJECTION),
      weightScales(buffers, config, QUERY_PROJECTION),
      bias(buffers, config, QUERY_PROJECTION), buffers.query, config);
  projectInt16Int8(
      buffers.input_int16, input_scale,
      matrixInt8(buffers, config, KEY_PROJECTION),
      weightScales(buffers, config, KEY_PROJECTION),
      bias(buffers, config, KEY_PROJECTION), buffers.key, config);
  projectInt16Int8(
      buffers.input_int16, input_scale,
      matrixInt8(buffers, config, VALUE_PROJECTION),
      weightScales(buffers, config, VALUE_PROJECTION),
      bias(buffers, config, VALUE_PROJECTION), buffers.value, config);
}

}  // namespace

bool endToEndTokenIsValid(uint16_t token) {
  return token % 7 != 5;
}

float endToEndFixtureInput(uint16_t token, uint16_t feature) {
  const int32_t value =
      static_cast<int32_t>((token * 37u + feature * 17u +
                            token * feature * 3u + 13u) %
                           101u) -
      50;
  return static_cast<float>(value) / 100.0f;
}

float endToEndFixtureWeight(uint8_t matrix_index,
                            uint16_t input_feature,
                            uint16_t output_feature) {
  const uint32_t mixed = (static_cast<uint32_t>(matrix_index) + 1u) * 29u +
                         input_feature * 31u + output_feature * 17u +
                         input_feature * output_feature * 3u;
  const int32_t value = static_cast<int32_t>(mixed % 127u) - 63;
  return static_cast<float>(value) / 256.0f;
}

float endToEndFixtureBias(uint8_t projection, uint16_t feature) {
  const int32_t value = static_cast<int32_t>(
                            ((static_cast<uint32_t>(projection) + 1u) * 11u +
                             feature * 7u) %
                            31u) -
                        15;
  return static_cast<float>(value) / 512.0f;
}

bool allocateEndToEndBuffers(const EndToEndConfig &config,
                             EndToEndBuffers &buffers) {
  if (config.heads == 0 || config.model_dimension % config.heads != 0 ||
      config.tile_size == 0) {
    return false;
  }
  const size_t activations = activationElements(config);
  const size_t matrices = matrixElements(config);
  const uint16_t head_dimension = config.model_dimension / config.heads;
  buffers.input = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.weights =
      static_cast<float *>(malloc(matrices * 4 * sizeof(float)));
  buffers.biases = static_cast<float *>(
      malloc(static_cast<size_t>(config.model_dimension) * 4 * sizeof(float)));
  buffers.query = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.key = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.value = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.context = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.reference = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.candidate = static_cast<float *>(malloc(activations * sizeof(float)));
  buffers.reference_scores =
      static_cast<float *>(malloc(config.sequence * sizeof(float)));
  buffers.scratch_scores = static_cast<float *>(
      malloc(static_cast<size_t>(config.tile_size) * 2 * sizeof(float)));
  buffers.scratch_value =
      static_cast<float *>(malloc(head_dimension * sizeof(float)));
  buffers.query_int8 = static_cast<int8_t *>(malloc(activations));
  buffers.key_int8 = static_cast<int8_t *>(malloc(activations));
  buffers.value_int16 =
      static_cast<int16_t *>(malloc(activations * sizeof(int16_t)));
  buffers.input_int16 =
      static_cast<int16_t *>(malloc(activations * sizeof(int16_t)));
  buffers.context_int16 =
      static_cast<int16_t *>(malloc(activations * sizeof(int16_t)));
  buffers.weights_int8 =
      static_cast<int8_t *>(malloc(matrices * 4 * sizeof(int8_t)));
  buffers.weight_scales = static_cast<float *>(
      malloc(static_cast<size_t>(config.model_dimension) * 4 * sizeof(float)));

  const bool allocated = buffers.input && buffers.weights && buffers.biases &&
                         buffers.query && buffers.key && buffers.value &&
                         buffers.context && buffers.reference &&
                         buffers.candidate && buffers.reference_scores &&
                         buffers.scratch_scores && buffers.scratch_value &&
                         buffers.query_int8 && buffers.key_int8 &&
                         buffers.value_int16 && buffers.input_int16 &&
                         buffers.context_int16 && buffers.weights_int8 &&
                         buffers.weight_scales;
  if (!allocated) {
    releaseEndToEndBuffers(buffers);
  }
  return allocated;
}

void releaseEndToEndBuffers(EndToEndBuffers &buffers) {
  free(buffers.input);
  free(buffers.weights);
  free(buffers.biases);
  free(buffers.query);
  free(buffers.key);
  free(buffers.value);
  free(buffers.context);
  free(buffers.reference);
  free(buffers.candidate);
  free(buffers.reference_scores);
  free(buffers.scratch_scores);
  free(buffers.scratch_value);
  free(buffers.query_int8);
  free(buffers.key_int8);
  free(buffers.value_int16);
  free(buffers.input_int16);
  free(buffers.context_int16);
  free(buffers.weights_int8);
  free(buffers.weight_scales);
  buffers = EndToEndBuffers{};
}

void initializeEndToEndFixture(const EndToEndConfig &config,
                               EndToEndBuffers &buffers) {
  for (uint16_t token = 0; token < config.sequence; ++token) {
    for (uint16_t feature = 0; feature < config.model_dimension; ++feature) {
      buffers.input[static_cast<size_t>(token) * config.model_dimension +
                    feature] = endToEndFixtureInput(token, feature);
    }
  }
  for (uint8_t projection = 0; projection < 4; ++projection) {
    float *projection_matrix = matrix(buffers, config, projection);
    float *projection_bias = bias(buffers, config, projection);
    for (uint16_t input_feature = 0;
         input_feature < config.model_dimension; ++input_feature) {
      for (uint16_t output_feature = 0;
           output_feature < config.model_dimension; ++output_feature) {
        projection_matrix[static_cast<size_t>(input_feature) *
                              config.model_dimension +
                          output_feature] =
            endToEndFixtureWeight(projection, input_feature, output_feature);
      }
    }
    for (uint16_t feature = 0; feature < config.model_dimension; ++feature) {
      projection_bias[feature] = endToEndFixtureBias(projection, feature);
    }
    quantizeWeightColumns(
        projection_matrix, matrixInt8(buffers, config, projection),
        weightScales(buffers, config, projection), config.model_dimension);
  }
}

void runEndToEndFloatReference(const EndToEndConfig &config,
                               EndToEndBuffers &buffers,
                               bool causal) {
  runProjections(config, buffers);
  attentionFloat(config, buffers, causal);
  project(buffers.context, matrix(buffers, config, OUTPUT_PROJECTION),
          bias(buffers, config, OUTPUT_PROJECTION), buffers.reference, config);
  zeroPaddedRows(buffers.reference, config);
}

void runEndToEndMixedTiled(const EndToEndConfig &config,
                           EndToEndBuffers &buffers,
                           bool causal) {
  runProjections(config, buffers);
  const size_t elements = activationElements(config);
  const float query_scale =
      quantizeInt8(buffers.query, buffers.query_int8, elements);
  const float key_scale = quantizeInt8(buffers.key, buffers.key_int8, elements);
  const float value_scale =
      quantizeInt16(buffers.value, buffers.value_int16, elements);
  attentionMixedTiled(config, buffers, causal, query_scale, key_scale,
                      value_scale);
  project(buffers.context, matrix(buffers, config, OUTPUT_PROJECTION),
          bias(buffers, config, OUTPUT_PROJECTION), buffers.candidate, config);
  zeroPaddedRows(buffers.candidate, config);
}

void runEndToEndIntProjectionMixedTiled(const EndToEndConfig &config,
                                        EndToEndBuffers &buffers,
                                        bool causal) {
  const size_t elements = activationElements(config);
  const float input_scale =
      quantizeInt16(buffers.input, buffers.input_int16, elements);
  runIntProjections(config, buffers, input_scale);
  const float query_scale =
      quantizeInt8(buffers.query, buffers.query_int8, elements);
  const float key_scale = quantizeInt8(buffers.key, buffers.key_int8, elements);
  const float value_scale =
      quantizeInt16(buffers.value, buffers.value_int16, elements);
  attentionMixedTiled(config, buffers, causal, query_scale, key_scale,
                      value_scale);
  const float context_scale =
      quantizeInt16(buffers.context, buffers.context_int16, elements);
  projectInt16Int8(
      buffers.context_int16, context_scale,
      matrixInt8(buffers, config, OUTPUT_PROJECTION),
      weightScales(buffers, config, OUTPUT_PROJECTION),
      bias(buffers, config, OUTPUT_PROJECTION), buffers.candidate, config);
  zeroPaddedRows(buffers.candidate, config);
}

size_t endToEndWeightBytes(const EndToEndConfig &config) {
  return (matrixElements(config) * 4 +
          static_cast<size_t>(config.model_dimension) * 4) *
         sizeof(float);
}

size_t endToEndMixedWorkspaceBytes(const EndToEndConfig &config) {
  const size_t head_dimension = config.model_dimension / config.heads;
  return (static_cast<size_t>(config.tile_size) * 2 + head_dimension) *
         sizeof(float);
}

size_t endToEndMixedWorkingSetBytes(const EndToEndConfig &config) {
  const size_t elements = activationElements(config);
  const size_t float_activations = elements * 6 * sizeof(float);
  const size_t quantized_activations = elements * 4;
  return endToEndWeightBytes(config) + float_activations +
         quantized_activations + endToEndMixedWorkspaceBytes(config);
}

size_t endToEndIntProjectionWeightBytes(const EndToEndConfig &config) {
  return matrixElements(config) * 4 * sizeof(int8_t) +
         static_cast<size_t>(config.model_dimension) * 4 *
             (sizeof(float) + sizeof(float));  // scales and biases
}

size_t endToEndIntProjectionWorkingSetBytes(const EndToEndConfig &config) {
  const size_t elements = activationElements(config);
  const size_t float_activations = elements * 6 * sizeof(float);
  const size_t quantized_attention = elements * 4;
  const size_t int16_projection_activations = elements * 2 * sizeof(int16_t);
  return endToEndIntProjectionWeightBytes(config) + float_activations +
         quantized_attention + int16_projection_activations +
         endToEndMixedWorkspaceBytes(config);
}
