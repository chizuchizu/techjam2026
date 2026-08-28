#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>

#include "tiny_transformer.h"

namespace {

using namespace tiny_transformer_weights;

constexpr uint8_t TIMING_SAMPLES = 7;
constexpr uint8_t GENERATION_STEPS = 48;

TinyTransformerBuffers buffers;
float logits[VOCABULARY];

void sortTimings(uint32_t *values, uint8_t count) {
  for (uint8_t i = 1; i < count; ++i) {
    const uint32_t value = values[i];
    int8_t j = i - 1;
    while (j >= 0 && values[j] > value) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = value;
  }
}

void benchmarkPrompt() {
  uint8_t tokens[CONTEXT];
  memcpy(tokens, PROMPT_TOKENS, sizeof(tokens));
  runTinyTransformer(tokens, logits, buffers);

  uint32_t timings[TIMING_SAMPLES];
  for (uint8_t sample = 0; sample < TIMING_SAMPLES; ++sample) {
    const int64_t begin = esp_timer_get_time();
    runTinyTransformer(tokens, logits, buffers);
    timings[sample] = static_cast<uint32_t>(esp_timer_get_time() - begin);
  }
  sortTimings(timings, TIMING_SAMPLES);

  const uint8_t predicted = tinyTransformerArgmax(logits);
  const int16_t expected = tinyTransformerTokenForByte('u');
  const bool passed = expected >= 0 && predicted == expected;
  Serial.println("TINY_TRANSFORMER_V1");
  Serial.printf("TINY_CONFIG,%u,%u,%u,%u,%u,%u\n", CONTEXT,
                MODEL_DIMENSION, HEADS, LAYERS, FFN_DIMENSION, VOCABULARY);
  Serial.printf("TINY_RESULT,%lu,%lu,%lu,%u,%u,%u,%s\n",
                static_cast<unsigned long>(timings[TIMING_SAMPLES / 2]),
                static_cast<unsigned long>(timings[0]),
                static_cast<unsigned long>(timings[TIMING_SAMPLES - 1]),
                static_cast<unsigned>(tinyTransformerWeightBytes()),
                static_cast<unsigned>(tinyTransformerWorkingSetBytes()),
                static_cast<unsigned>(predicted), passed ? "PASS" : "FAIL");
  for (uint8_t token = 0; token < VOCABULARY; ++token) {
    Serial.printf("TINY_LOGIT,%u,%.9g\n", token, logits[token]);
  }
  Serial.println("TINY_DONE");
}

void generateText() {
  uint8_t tokens[CONTEXT];
  memcpy(tokens, PROMPT_TOKENS, sizeof(tokens));
  uint64_t total_us = 0;
  Serial.println("TINY_GENERATION_V1");
  for (uint8_t step = 0; step < GENERATION_STEPS; ++step) {
    const int64_t begin = esp_timer_get_time();
    runTinyTransformer(tokens, logits, buffers);
    const uint32_t elapsed = static_cast<uint32_t>(esp_timer_get_time() - begin);
    total_us += elapsed;
    const uint8_t predicted = tinyTransformerArgmax(logits);
    Serial.printf("TINY_GENERATED,%u,%u,%lu\n", step, predicted,
                  static_cast<unsigned long>(elapsed));
    memmove(tokens, tokens + 1, CONTEXT - 1);
    tokens[CONTEXT - 1] = predicted;
  }
  Serial.printf("TINY_GENERATION_DONE,%llu\n",
                static_cast<unsigned long long>(total_us));
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("XIAO_ESP32C3_TINY_TRANSFORMER");
  if (!allocateTinyTransformerBuffers(buffers)) {
    Serial.printf("TINY_ALLOCATION_FAILED,%u\n",
                  static_cast<unsigned>(ESP.getFreeHeap()));
    return;
  }
  Serial.printf("TINY_READY,%u,%u\n",
                static_cast<unsigned>(tinyTransformerWeightBytes()),
                static_cast<unsigned>(tinyTransformerWorkingSetBytes()));
  Serial.println("Send f for validated inference or g for generation.");
}

void loop() {
  if (!Serial.available() || !buffers.values) {
    delay(1);
    return;
  }
  const char command = static_cast<char>(Serial.read());
  if (command == 'f' || command == 'F') benchmarkPrompt();
  if (command == 'g' || command == 'G') generateText();
}
