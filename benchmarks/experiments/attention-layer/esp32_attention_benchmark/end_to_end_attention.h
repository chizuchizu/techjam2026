#pragma once

#include <stddef.h>
#include <stdint.h>

struct EndToEndConfig {
  uint16_t sequence;
  uint16_t model_dimension;
  uint16_t heads;
  uint16_t tile_size;
};

struct EndToEndBuffers {
  float *input = nullptr;
  float *weights = nullptr;
  float *biases = nullptr;
  float *query = nullptr;
  float *key = nullptr;
  float *value = nullptr;
  float *context = nullptr;
  float *reference = nullptr;
  float *candidate = nullptr;
  float *reference_scores = nullptr;
  float *scratch_scores = nullptr;
  float *scratch_value = nullptr;
  int8_t *query_int8 = nullptr;
  int8_t *key_int8 = nullptr;
  int16_t *value_int16 = nullptr;
  int16_t *input_int16 = nullptr;
  int16_t *context_int16 = nullptr;
  int8_t *weights_int8 = nullptr;
  float *weight_scales = nullptr;
};

// The fixture uses formulas rather than an on-device PRNG so the host validator
// can independently reconstruct every input, weight, bias, and padding bit.
bool endToEndTokenIsValid(uint16_t token);
float endToEndFixtureInput(uint16_t token, uint16_t feature);
float endToEndFixtureWeight(uint8_t matrix,
                            uint16_t input_feature,
                            uint16_t output_feature);
float endToEndFixtureBias(uint8_t projection, uint16_t feature);

bool allocateEndToEndBuffers(const EndToEndConfig &config,
                             EndToEndBuffers &buffers);
void releaseEndToEndBuffers(EndToEndBuffers &buffers);
void initializeEndToEndFixture(const EndToEndConfig &config,
                               EndToEndBuffers &buffers);

// Full float reference: Q/K/V projections, masked multi-head attention, and
// output projection. The reference materializes one score row at a time.
void runEndToEndFloatReference(const EndToEndConfig &config,
                               EndToEndBuffers &buffers,
                               bool causal);

// Candidate path: the same projections, int8 Q/K + int16 V quantization,
// tiled online softmax, fused V dequantization, and output projection.
void runEndToEndMixedTiled(const EndToEndConfig &config,
                           EndToEndBuffers &buffers,
                           bool causal);

// Projection-optimized candidate: float inputs are converted to int16, all
// four projection matrices use per-output-channel int8 weights and int32 dot
// products, and the selected mixed tiled attention kernel remains unchanged.
void runEndToEndIntProjectionMixedTiled(const EndToEndConfig &config,
                                        EndToEndBuffers &buffers,
                                        bool causal);

size_t endToEndWeightBytes(const EndToEndConfig &config);
size_t endToEndMixedWorkspaceBytes(const EndToEndConfig &config);
size_t endToEndMixedWorkingSetBytes(const EndToEndConfig &config);
size_t endToEndIntProjectionWeightBytes(const EndToEndConfig &config);
size_t endToEndIntProjectionWorkingSetBytes(const EndToEndConfig &config);
