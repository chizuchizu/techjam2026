#include "attention_kernels.h"

#include <float.h>
#include <math.h>
#include <string.h>

namespace {

inline uint16_t validKeysForQuery(uint16_t query_index,
                                  const AttentionShape &shape) {
  return shape.causal ? query_index + 1 : shape.sequence;
}

inline float dotProduct(const float *left, const float *right, uint16_t count) {
  float sum = 0.0f;
  for (uint16_t index = 0; index < count; ++index) {
    sum += left[index] * right[index];
  }
  return sum;
}

inline int32_t dotProductInt8(const int8_t *left,
                             const int8_t *right,
                             uint16_t count) {
  int32_t sum = 0;
  for (uint16_t index = 0; index < count; ++index) {
    sum += static_cast<int32_t>(left[index]) * right[index];
  }
  return sum;
}

// Range-reduced fifth-order approximation. Softmax only calls exp(x) for
// x <= 0. On this soft-float ESP32-C3, avoiding libm's expf is valuable. The
// benchmark checks the resulting attention output against the exact reference.
inline float fastExpNegative(float x) {
  if (x <= -87.0f) {
    return 0.0f;
  }
  if (x > 0.0f) {
    x = 0.0f;
  }

  constexpr float inverse_ln2 = 1.4426950408889634f;
  constexpr float ln2 = 0.6931471805599453f;
  const int exponent = static_cast<int>(floorf(x * inverse_ln2 + 0.5f));
  const float remainder = x - static_cast<float>(exponent) * ln2;
  const float r2 = remainder * remainder;
  const float polynomial =
      1.0f + remainder + r2 *
      (0.5f + remainder *
      (0.1666666716f + remainder *
      (0.0416666679f + remainder * 0.0083333338f)));

  if (exponent < -126) {
    return 0.0f;
  }
  union {
    uint32_t bits;
    float value;
  } power_of_two = {
      static_cast<uint32_t>(exponent + 127) << 23,
  };
  return polynomial * power_of_two.value;
}

template <bool use_fast_exp>
inline float selectedExp(float x) {
  if constexpr (use_fast_exp) {
    return fastExpNegative(x);
  }
  return expf(x);
}

template <bool use_fast_exp>
void attentionTiledOnline(const float *query,
                          const float *key,
                          const float *value,
                          float *output,
                          float *scratch_scores,
                          float *scratch_value,
                          uint16_t tile_size,
                          const AttentionShape &shape) {
  const uint16_t sequence = shape.sequence;
  const uint16_t dimension = shape.head_dimension;
  const float scale = 1.0f / sqrtf(static_cast<float>(dimension));

  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    const float *query_row = query + static_cast<size_t>(query_index) * dimension;
    float *output_row = output + static_cast<size_t>(query_index) * dimension;
    memset(output_row, 0, static_cast<size_t>(dimension) * sizeof(float));

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    bool first_tile = true;
    const uint16_t valid_keys = validKeysForQuery(query_index, shape);

    for (uint16_t tile_begin = 0; tile_begin < valid_keys;
         tile_begin += tile_size) {
      const uint16_t tile_end =
          static_cast<uint16_t>(tile_begin + tile_size < valid_keys
                                    ? tile_begin + tile_size
                                    : valid_keys);
      const uint16_t tile_count = tile_end - tile_begin;
      float tile_max = -FLT_MAX;

      for (uint16_t offset = 0; offset < tile_count; ++offset) {
        const uint16_t key_index = tile_begin + offset;
        const float *key_row = key + static_cast<size_t>(key_index) * dimension;
        const float score = dotProduct(query_row, key_row, dimension) * scale;
        scratch_scores[offset] = score;
        if (score > tile_max) {
          tile_max = score;
        }
      }

      memset(scratch_value, 0, static_cast<size_t>(dimension) * sizeof(float));
      float tile_sum = 0.0f;
      for (uint16_t offset = 0; offset < tile_count; ++offset) {
        const uint16_t key_index = tile_begin + offset;
        const float weight = selectedExp<use_fast_exp>(scratch_scores[offset] - tile_max);
        tile_sum += weight;
        const float *value_row = value + static_cast<size_t>(key_index) * dimension;
        for (uint16_t feature = 0; feature < dimension; ++feature) {
          scratch_value[feature] += weight * value_row[feature];
        }
      }

      if (first_tile) {
        memcpy(output_row, scratch_value,
               static_cast<size_t>(dimension) * sizeof(float));
        running_max = tile_max;
        running_sum = tile_sum;
        first_tile = false;
        continue;
      }

      const float merged_max = tile_max > running_max ? tile_max : running_max;
      const float old_scale = selectedExp<use_fast_exp>(running_max - merged_max);
      const float tile_scale = selectedExp<use_fast_exp>(tile_max - merged_max);
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        output_row[feature] = output_row[feature] * old_scale +
                              scratch_value[feature] * tile_scale;
      }
      running_sum = running_sum * old_scale + tile_sum * tile_scale;
      running_max = merged_max;
    }

    const float inverse_sum = 1.0f / running_sum;
    for (uint16_t feature = 0; feature < dimension; ++feature) {
      output_row[feature] *= inverse_sum;
    }
  }
}

}  // namespace

void attentionMaterializedReference(const float *query,
                                    const float *key,
                                    const float *value,
                                    float *output,
                                    float *scores,
                                    const AttentionShape &shape) {
  const uint16_t sequence = shape.sequence;
  const uint16_t dimension = shape.head_dimension;
  const float scale = 1.0f / sqrtf(static_cast<float>(dimension));

  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    const float *query_row = query + static_cast<size_t>(query_index) * dimension;
    float *score_row = scores + static_cast<size_t>(query_index) * sequence;
    const uint16_t valid_keys = validKeysForQuery(query_index, shape);
    float maximum = -FLT_MAX;

    for (uint16_t key_index = 0; key_index < valid_keys; ++key_index) {
      const float *key_row = key + static_cast<size_t>(key_index) * dimension;
      const float score = dotProduct(query_row, key_row, dimension) * scale;
      score_row[key_index] = score;
      if (score > maximum) {
        maximum = score;
      }
    }

    float sum = 0.0f;
    for (uint16_t key_index = 0; key_index < valid_keys; ++key_index) {
      const float probability = expf(score_row[key_index] - maximum);
      score_row[key_index] = probability;
      sum += probability;
    }
    const float inverse_sum = 1.0f / sum;

    float *output_row = output + static_cast<size_t>(query_index) * dimension;
    memset(output_row, 0, static_cast<size_t>(dimension) * sizeof(float));
    for (uint16_t key_index = 0; key_index < valid_keys; ++key_index) {
      const float probability = score_row[key_index] * inverse_sum;
      const float *value_row = value + static_cast<size_t>(key_index) * dimension;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        output_row[feature] += probability * value_row[feature];
      }
    }
  }
}

void attentionTiledOnlineExact(const float *query,
                               const float *key,
                               const float *value,
                               float *output,
                               float *scratch_scores,
                               float *scratch_value,
                               uint16_t tile_size,
                               const AttentionShape &shape) {
  attentionTiledOnline<false>(query, key, value, output, scratch_scores,
                              scratch_value, tile_size, shape);
}

void attentionTiledOnlineFastExp(const float *query,
                                 const float *key,
                                 const float *value,
                                 float *output,
                                 float *scratch_scores,
                                 float *scratch_value,
                                 uint16_t tile_size,
                                 const AttentionShape &shape) {
  attentionTiledOnline<true>(query, key, value, output, scratch_scores,
                             scratch_value, tile_size, shape);
}

template <typename ValueType>
void attentionMaterializedQuantized(const int8_t *query,
                                    const int8_t *key,
                                    const ValueType *value,
                                    const QuantizationScales &scales,
                                    float *output,
                                    float *scores,
                                    const AttentionShape &shape) {
  const uint16_t sequence = shape.sequence;
  const uint16_t dimension = shape.head_dimension;
  const float score_scale = scales.query * scales.key /
                            sqrtf(static_cast<float>(dimension));

  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    const int8_t *query_row =
        query + static_cast<size_t>(query_index) * dimension;
    float *score_row = scores + static_cast<size_t>(query_index) * sequence;
    const uint16_t valid_keys = validKeysForQuery(query_index, shape);
    float maximum = -FLT_MAX;

    for (uint16_t key_index = 0; key_index < valid_keys; ++key_index) {
      const int8_t *key_row =
          key + static_cast<size_t>(key_index) * dimension;
      const float score = static_cast<float>(
                              dotProductInt8(query_row, key_row, dimension)) *
                          score_scale;
      score_row[key_index] = score;
      if (score > maximum) {
        maximum = score;
      }
    }

    float sum = 0.0f;
    for (uint16_t key_index = 0; key_index < valid_keys; ++key_index) {
      const float probability = expf(score_row[key_index] - maximum);
      score_row[key_index] = probability;
      sum += probability;
    }
    const float output_scale = scales.value / sum;

    float *output_row = output + static_cast<size_t>(query_index) * dimension;
    memset(output_row, 0, static_cast<size_t>(dimension) * sizeof(float));
    for (uint16_t key_index = 0; key_index < valid_keys; ++key_index) {
      const float probability = score_row[key_index];
      const ValueType *value_row =
          value + static_cast<size_t>(key_index) * dimension;
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        output_row[feature] += probability * value_row[feature];
      }
    }
    for (uint16_t feature = 0; feature < dimension; ++feature) {
      output_row[feature] *= output_scale;
    }
  }
}

template <typename ValueType>
void attentionTiledOnlineQuantized(const int8_t *query,
                                   const int8_t *key,
                                   const ValueType *value,
                                   const QuantizationScales &scales,
                                   float *output,
                                   float *scratch_scores,
                                   float *scratch_value,
                                   uint16_t tile_size,
                                   const AttentionShape &shape) {
  const uint16_t sequence = shape.sequence;
  const uint16_t dimension = shape.head_dimension;
  const float score_scale = scales.query * scales.key /
                            sqrtf(static_cast<float>(dimension));

  for (uint16_t query_index = 0; query_index < sequence; ++query_index) {
    const int8_t *query_row =
        query + static_cast<size_t>(query_index) * dimension;
    float *output_row = output + static_cast<size_t>(query_index) * dimension;
    memset(output_row, 0, static_cast<size_t>(dimension) * sizeof(float));
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    bool first_tile = true;
    const uint16_t valid_keys = validKeysForQuery(query_index, shape);

    for (uint16_t tile_begin = 0; tile_begin < valid_keys;
         tile_begin += tile_size) {
      const uint16_t tile_end =
          static_cast<uint16_t>(tile_begin + tile_size < valid_keys
                                    ? tile_begin + tile_size
                                    : valid_keys);
      const uint16_t tile_count = tile_end - tile_begin;
      float tile_max = -FLT_MAX;

      for (uint16_t offset = 0; offset < tile_count; ++offset) {
        const uint16_t key_index = tile_begin + offset;
        const int8_t *key_row =
            key + static_cast<size_t>(key_index) * dimension;
        const float score = static_cast<float>(
                                dotProductInt8(query_row, key_row, dimension)) *
                            score_scale;
        scratch_scores[offset] = score;
        if (score > tile_max) {
          tile_max = score;
        }
      }

      memset(scratch_value, 0, static_cast<size_t>(dimension) * sizeof(float));
      float tile_sum = 0.0f;
      for (uint16_t offset = 0; offset < tile_count; ++offset) {
        const uint16_t key_index = tile_begin + offset;
        const float weight = expf(scratch_scores[offset] - tile_max);
        tile_sum += weight;
        const ValueType *value_row =
            value + static_cast<size_t>(key_index) * dimension;
        for (uint16_t feature = 0; feature < dimension; ++feature) {
          scratch_value[feature] += weight * value_row[feature];
        }
      }

      if (first_tile) {
        memcpy(output_row, scratch_value,
               static_cast<size_t>(dimension) * sizeof(float));
        running_max = tile_max;
        running_sum = tile_sum;
        first_tile = false;
        continue;
      }

      const float merged_max = tile_max > running_max ? tile_max : running_max;
      const float old_scale = expf(running_max - merged_max);
      const float tile_scale = expf(tile_max - merged_max);
      for (uint16_t feature = 0; feature < dimension; ++feature) {
        output_row[feature] = output_row[feature] * old_scale +
                              scratch_value[feature] * tile_scale;
      }
      running_sum = running_sum * old_scale + tile_sum * tile_scale;
      running_max = merged_max;
    }

    const float output_scale = scales.value / running_sum;
    for (uint16_t feature = 0; feature < dimension; ++feature) {
      output_row[feature] *= output_scale;
    }
  }
}

void attentionMaterializedInt8(const int8_t *query,
                               const int8_t *key,
                               const int8_t *value,
                               const QuantizationScales &scales,
                               float *output,
                               float *scores,
                               const AttentionShape &shape) {
  attentionMaterializedQuantized(query, key, value, scales, output, scores,
                                 shape);
}

void attentionTiledOnlineInt8(const int8_t *query,
                              const int8_t *key,
                              const int8_t *value,
                              const QuantizationScales &scales,
                              float *output,
                              float *scratch_scores,
                              float *scratch_value,
                              uint16_t tile_size,
                              const AttentionShape &shape) {
  attentionTiledOnlineQuantized(query, key, value, scales, output,
                                scratch_scores, scratch_value, tile_size,
                                shape);
}

void attentionMaterializedInt8QKInt16V(const int8_t *query,
                                       const int8_t *key,
                                       const int16_t *value,
                                       const QuantizationScales &scales,
                                       float *output,
                                       float *scores,
                                       const AttentionShape &shape) {
  attentionMaterializedQuantized(query, key, value, scales, output, scores,
                                 shape);
}

void attentionTiledOnlineInt8QKInt16V(const int8_t *query,
                                      const int8_t *key,
                                      const int16_t *value,
                                      const QuantizationScales &scales,
                                      float *output,
                                      float *scratch_scores,
                                      float *scratch_value,
                                      uint16_t tile_size,
                                      const AttentionShape &shape) {
  attentionTiledOnlineQuantized(query, key, value, scales, output,
                                scratch_scores, scratch_value, tile_size,
                                shape);
}

AccuracyStats compareAttentionOutputs(const float *reference,
                                      const float *candidate,
                                      size_t elements,
                                      float relative_tolerance,
                                      float absolute_tolerance) {
  AccuracyStats stats = {0.0f, 0.0f, 0, static_cast<uint32_t>(elements)};
  for (size_t index = 0; index < elements; ++index) {
    const float absolute_error = fabsf(candidate[index] - reference[index]);
    const float reference_magnitude = fabsf(reference[index]);
    const float relative_error =
        absolute_error / (reference_magnitude > 1.0e-12f
                              ? reference_magnitude
                              : 1.0e-12f);
    if (absolute_error > stats.max_absolute_error) {
      stats.max_absolute_error = absolute_error;
    }
    if (relative_error > stats.max_relative_error) {
      stats.max_relative_error = relative_error;
    }
    const bool absolute_ok = absolute_error <= absolute_tolerance;
    const bool relative_ok = absolute_error <=
                             relative_tolerance * reference_magnitude;
    if (!absolute_ok && !relative_ok) {
      ++stats.failed_elements;
    }
  }
  return stats;
}

size_t materializedWorkspaceBytes(const AttentionShape &shape) {
  return static_cast<size_t>(shape.sequence) * shape.sequence * sizeof(float);
}

size_t tiledOnlineWorkspaceBytes(const AttentionShape &shape,
                                 uint16_t tile_size) {
  return (static_cast<size_t>(shape.head_dimension) + tile_size) * sizeof(float);
}

size_t materializedInt8WorkingSetBytes(const AttentionShape &shape) {
  const size_t elements =
      static_cast<size_t>(shape.sequence) * shape.head_dimension;
  return elements * 3 + elements * sizeof(float) +
         materializedWorkspaceBytes(shape);
}

size_t tiledOnlineInt8WorkingSetBytes(const AttentionShape &shape,
                                      uint16_t tile_size) {
  const size_t elements =
      static_cast<size_t>(shape.sequence) * shape.head_dimension;
  return elements * 3 + elements * sizeof(float) +
         tiledOnlineWorkspaceBytes(shape, tile_size);
}

size_t materializedInt8QKInt16VWorkingSetBytes(const AttentionShape &shape) {
  const size_t elements =
      static_cast<size_t>(shape.sequence) * shape.head_dimension;
  return elements * 4 + elements * sizeof(float) +
         materializedWorkspaceBytes(shape);
}

size_t tiledOnlineInt8QKInt16VWorkingSetBytes(const AttentionShape &shape,
                                              uint16_t tile_size) {
  const size_t elements =
      static_cast<size_t>(shape.sequence) * shape.head_dimension;
  return elements * 4 + elements * sizeof(float) +
         tiledOnlineWorkspaceBytes(shape, tile_size);
}
