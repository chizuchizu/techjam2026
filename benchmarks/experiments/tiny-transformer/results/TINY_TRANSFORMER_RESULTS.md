# Complete tiny Transformer result

The XIAO ESP32-C3 now runs a trained causal character Transformer end to end,
not only its attention kernel. The deployed graph contains token and positional
embeddings, two pre-norm blocks, four causal attention heads, residual paths,
two ReLU feed-forward networks, final RMSNorm, and a tied language-model head.

## Model and deployment

| Property | Value |
|---|---:|
| Context / model width | 16 / 32 |
| Heads / head width | 4 / 8 |
| Blocks / FFN width | 2 / 64 |
| Vocabulary | 24 characters |
| Trained parameters | 17,824 |
| Stored inference weights | 24,800 B |
| Runtime working set | 22,688 B |
| Compiled sketch flash | 305,686 B |
| Compiled static RAM | 13,512 B |

Linear matrices use per-output-channel int8 weights and dynamically quantized
int16 activations. Attention dynamically quantizes Q/K to int8 and V to int16,
uses int32 QK dot products, and applies stable causal float softmax. Weight
quantization happens offline and is excluded from inference timing; activation
quantization is included.

## Physical-board result

The raw capture is
[`esp32c3_tiny_transformer_v1.log`](esp32c3_tiny_transformer_v1.log).

| Measurement | Result |
|---|---:|
| Median complete forward | 106.614 ms |
| Minimum / maximum of 7 | 106.596 / 106.636 ms |
| 48-token generation | 5.118654 s |
| Mean generated-token time | 106.639 ms |
| Approximate generation rate | 9.38 tokens/s |

The independent NumPy deployment implementation passes all 126 next-token
windows in the training corpus. Against the physical board, all 24 prompt
logits pass the project gate; the maximum absolute difference is 0.0004802 and
all 48 greedy-generated token IDs match exactly.

This is a deployment proof, not a language-quality claim. The model was trained
to memorize a 126-character corpus, so its 100% corpus-window accuracy does not
measure held-out generalization.
