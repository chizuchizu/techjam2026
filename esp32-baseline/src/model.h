/*
 * model.h - the case-2 transformer forward pass.
 */
#ifndef TM_MODEL_H
#define TM_MODEL_H

#include "tm_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parsed Q12 weight blob (see tm_config.h for the on-disk layout). */
typedef struct {
    const int16_t* q[TM_L][6];   /* q,k,v,o,f1,f2 weight matrices  */
    float          ws[TM_L][6];  /* per-matrix max|W|/2047         */
} TMQ12Weights;

void tm_scan_q12(const void* blob, TMQ12Weights* out);

/* Set numeric mode (TM_MODE_EXACT / TM_MODE_FAST). */
void tm_set_mode(int mode);

int  tm_get_mode(void);

/* Forward: y = model(x), x,y: [TM_S*TM_D] fp32 row-major.
 * weights_f32: flat fp32 buffer (TM_W_TOTAL floats).
 * q12: parsed Q12 blob (may be NULL when mode==EXACT).
 */
void tm_forward(const float* x, float* y,
                const float* weights_f32, const TMQ12Weights* q12);

#ifdef __cplusplus
}
#endif

/* workspace accessors (device firmware reads input / writes output
 * directly into the static arena to avoid a 128 KB duplicate) */
float* tm_input(void);
float* tm_output(void);

#endif /* TM_MODEL_H */
