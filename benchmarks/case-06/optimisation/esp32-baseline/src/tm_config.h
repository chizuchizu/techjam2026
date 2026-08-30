/*
 * tm_config.h - Tech Jam transformer configuration for the ESP32-C3.
 *
 * Target: Seeed XIAO ESP32C3 (400 KB SRAM, 4 MB flash, 160 MHz RV32IMC, NO FPU).
 * Model : B=10000 (informational; the firmware still runs one forward per
 *         input frame and the B=10000 batch is streamed frame-by-frame),
 *         S/D/H/F/L set below (case 6 defaults S=128 D=128 H=4 F=128 L=4),
 *         causal, fp32 baseline.
 *
 * The forward pass has two selectable numeric modes:
 *   TM_MODE_EXACT - fp32 GEMM/reference arithmetic with shared quantized
 *                   attention staging (reference-quality, not bit-exact).
 *   TM_MODE_FAST  - Q15xQ12 fixed-point GEMM, integer attention/exp lookup,
 *                   and fused quantization. (~5.27 s/forward; validated vs
 *                   the benchmark gate on host and physical C3 hardware.)
 * Both modes are always compiled; select at runtime with tm_set_mode().
 */
#ifndef TM_CONFIG_H
#define TM_CONFIG_H

#include <stdint.h>

/* ---------- model geometry (per-case; case 6 defaults) ---------- */
#define TM_B      10000   /* batch size (case 6): informational; firmware does one forward per input frame; the batch is streamed */
/* Per-case geometry — set to the target case. The Q15/Q12 asm GEMM kernels are
 * only compiled for K==128 (TM_D==128 && TM_F==128); other shapes fall back to
 * the validated C paths automatically (see kernels.c). */
#define TM_S      128
#define TM_D      128
#define TM_H      4
#define TM_HD     (TM_D / TM_H)          /* 32 */
#define TM_F      128
#define TM_L      4
#define TM_CAUSAL 1

#define TM_LN_EPS 1e-5f

/* score scale = 1/sqrt(head_dim) (fp32, matches torch BaselineSelfAttention).
 * Depends on TM_HD so head/dim sweeps stay correct; == 1/sqrt(32) for case 6. */
#define TM_ATTN_SCALE (1.0f / sqrtf((float)TM_HD))

/* ---------- numeric modes ---------- */
#define TM_MODE_EXACT 0
#define TM_MODE_FAST  1
#ifndef TM_MODE_DEFAULT
#define TM_MODE_DEFAULT TM_MODE_FAST
#endif

/* ---------- fixed-point GEMM (FAST mode) ---------- */
#define TM_QACT_BITS  15      /* activation quantization: [-32767,32767]      */
#define TM_QACT_MAX   32767.0f
#define TM_QWT_BITS   12      /* weight quantization: [-2047,2047]            */
#define TM_QWT_MAX    2047.0f

/* ---------- weight layout ---------- */
/* Flat fp32 weight buffer, per layer l in [0,TM_L):
 *   0 norm1.weight[D]  1 norm1.bias[D]
 *   2 q.weight[D*D]    3 q.bias[D]
 *   4 k.weight[D*D]    5 k.bias[D]
 *   6 v.weight[D*D]    7 v.bias[D]
 *   8 o.weight[D*D]    9 o.bias[D]
 *  10 norm2.weight[D] 11 norm2.bias[D]
 *  12 f1.weight[F*D]  13 f1.bias[F]
 *  14 f2.weight[D*F]  15 f2.bias[D]
 * then final_norm.weight[D], final_norm.bias[D].
 * Total = L*(2D + 4(D^2+D) + 2D + (FD+F) + (DF+D)) + 2D floats
 *         = L*TM_W_LAYER_FLOATS + 2*TM_D (matches the torch param count).
 */
#define TM_W_BLK_N1W 0
#define TM_W_BLK_N1B 1
#define TM_W_BLK_QW  2
#define TM_W_BLK_QB  3
#define TM_W_BLK_KW  4
#define TM_W_BLK_KB  5
#define TM_W_BLK_VW  6
#define TM_W_BLK_VB  7
#define TM_W_BLK_OW  8
#define TM_W_BLK_OB  9
#define TM_W_BLK_N2W 10
#define TM_W_BLK_N2B 11
#define TM_W_BLK_F1W 12
#define TM_W_BLK_F1B 13
#define TM_W_BLK_F2W 14
#define TM_W_BLK_F2B 15

#define TM_W_BLK_FLOATS(b) \
    ((b)==TM_W_BLK_N1W||(b)==TM_W_BLK_N1B||(b)==TM_W_BLK_N2W||(b)==TM_W_BLK_N2B ? TM_D : \
     (b)==TM_W_BLK_F1W   ? (TM_F*TM_D) : \
     (b)==TM_W_BLK_F1B   ? TM_F : \
     (b)==TM_W_BLK_F2W   ? (TM_D*TM_F) : \
     (b)==TM_W_BLK_F2B   ? TM_D : \
     (b)==TM_W_BLK_QB||(b)==TM_W_BLK_KB||(b)==TM_W_BLK_VB||(b)==TM_W_BLK_OB ? TM_D : \
     TM_D*TM_D)   /* q,k,v,o weight matrices */

#define TM_W_LAYER_FLOATS \
    (2*TM_D + 4*(TM_D*TM_D + TM_D) + 2*TM_D + (TM_F*TM_D + TM_F) + (TM_D*TM_F + TM_D))
#define TM_W_FINAL_OFF  ((uint32_t)TM_L * TM_W_LAYER_FLOATS)
#define TM_W_TOTAL      (TM_W_FINAL_OFF + 2*TM_D)   /* 398,592 */
#define TM_W_FINALW     (TM_W_FINAL_OFF + 0)
#define TM_W_FINALB     (TM_W_FINAL_OFF + TM_D)

/* Q12 weight blob layout (for FAST GEMM). For each of TM_L*6 linear matrices
 * in fixed order [layer][q,k,v,o,f1,f2]:
 *   uint32 count   = out*in
 *   float  w_scale = max|W| / 2047   (1 / quantization scale)
 *   int16  data[count]
 * Total bytes = layers*6*(8 + 2*count).
 */
#define TM_Q12_PER_LAYER_MATS 6

#endif /* TM_CONFIG_H */

/* R1 integer-residual span (model residual magnitude bound). */
#define TM_RES_SPAN 16.0f
