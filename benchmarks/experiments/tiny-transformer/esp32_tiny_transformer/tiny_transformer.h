#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tiny_transformer_weights.h"

struct TinyTransformerBuffers {
  float *values = nullptr;
  float *normalized = nullptr;
  float *query = nullptr;
  float *key = nullptr;
  float *value = nullptr;
  float *context = nullptr;
  float *projection = nullptr;
  float *hidden = nullptr;
  int16_t *activation_int16 = nullptr;
  int8_t *query_int8 = nullptr;
  int8_t *key_int8 = nullptr;
  int16_t *value_int16 = nullptr;
  float *scores = nullptr;
};

bool allocateTinyTransformerBuffers(TinyTransformerBuffers &buffers);
void releaseTinyTransformerBuffers(TinyTransformerBuffers &buffers);

void runTinyTransformer(const uint8_t *tokens,
                        float *logits,
                        TinyTransformerBuffers &buffers);

size_t tinyTransformerWeightBytes();
size_t tinyTransformerWorkingSetBytes();
uint8_t tinyTransformerArgmax(const float *logits);
int16_t tinyTransformerTokenForByte(uint8_t value);
