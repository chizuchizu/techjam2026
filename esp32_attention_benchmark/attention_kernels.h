#pragma once

#include <stddef.h>
#include <stdint.h>

struct AttentionShape {
  uint16_t sequence;
  uint16_t head_dimension;
  bool causal;
};

struct AccuracyStats {
  float max_absolute_error;
  float max_relative_error;
  uint32_t failed_elements;
  uint32_t total_elements;
};

struct QuantizationScales {
  float query;
  float key;
  float value;
};

// Reference scaled dot-product attention. The caller supplies sequence^2 floats
// for scores, so this deliberately demonstrates the quadratic-memory schedule.
void attentionMaterializedReference(const float *query,
                                    const float *key,
                                    const float *value,
                                    float *output,
                                    float *scores,
                                    const AttentionShape &shape);

// A FlashAttention-style, block-online softmax. Neither variant materializes
// the attention matrix. scratch_scores needs tile_size floats and scratch_value
// needs head_dimension floats.
void attentionTiledOnlineExact(const float *query,
                               const float *key,
                               const float *value,
                               float *output,
                               float *scratch_scores,
                               float *scratch_value,
                               uint16_t tile_size,
                               const AttentionShape &shape);

void attentionTiledOnlineFastExp(const float *query,
                                 const float *key,
                                 const float *value,
                                 float *output,
                                 float *scratch_scores,
                                 float *scratch_value,
                                 uint16_t tile_size,
                                 const AttentionShape &shape);

// Hybrid int8 kernels assume Q/K/V have already been symmetrically quantized.
// They use int32 dot products, stable float softmax, and a float output. This is
// representative of an end-to-end quantized model, so quantization is excluded
// from kernel timing but its error is included in the accuracy comparison.
void attentionMaterializedInt8(const int8_t *query,
                               const int8_t *key,
                               const int8_t *value,
                               const QuantizationScales &scales,
                               float *output,
                               float *scores,
                               const AttentionShape &shape);

void attentionTiledOnlineInt8(const int8_t *query,
                              const int8_t *key,
                              const int8_t *value,
                              const QuantizationScales &scales,
                              float *output,
                              float *scratch_scores,
                              float *scratch_value,
                              uint16_t tile_size,
                              const AttentionShape &shape);

void attentionMaterializedInt8QKInt16V(const int8_t *query,
                                       const int8_t *key,
                                       const int16_t *value,
                                       const QuantizationScales &scales,
                                       float *output,
                                       float *scores,
                                       const AttentionShape &shape);

void attentionTiledOnlineInt8QKInt16V(const int8_t *query,
                                      const int8_t *key,
                                      const int16_t *value,
                                      const QuantizationScales &scales,
                                      float *output,
                                      float *scratch_scores,
                                      float *scratch_value,
                                      uint16_t tile_size,
                                      const AttentionShape &shape);

AccuracyStats compareAttentionOutputs(const float *reference,
                                      const float *candidate,
                                      size_t elements,
                                      float relative_tolerance,
                                      float absolute_tolerance);

size_t materializedWorkspaceBytes(const AttentionShape &shape);
size_t tiledOnlineWorkspaceBytes(const AttentionShape &shape,
                                 uint16_t tile_size);

size_t materializedInt8WorkingSetBytes(const AttentionShape &shape);
size_t tiledOnlineInt8WorkingSetBytes(const AttentionShape &shape,
                                      uint16_t tile_size);

size_t materializedInt8QKInt16VWorkingSetBytes(const AttentionShape &shape);
size_t tiledOnlineInt8QKInt16VWorkingSetBytes(const AttentionShape &shape,
                                              uint16_t tile_size);
