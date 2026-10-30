#ifndef MLP_H
#define MLP_H
#include <stdint.h>

enum { MLP_INPUT = 784, MLP_HIDDEN = 128, MLP_OUTPUT = 10,
       MLP_FRAC = 6, MLP_BIAS_LIMIT = 16777216 };

/* Row-major signed weights; zero point = 0. Input has six fractional bits.
   Bias scales: b1=2^-(6+w1_frac), b2=2^-(hidden_frac+w2_frac). */
typedef struct {
    int8_t w1[MLP_HIDDEN * MLP_INPUT];
    int8_t w2[MLP_OUTPUT * MLP_HIDDEN];
    int32_t b1[MLP_HIDDEN];
    int32_t b2[MLP_OUTPUT];
    uint32_t w1_frac, hidden_frac, w2_frac;
} mlp_model;

/* Both dot functions require n <= 784 and buffers of at least n bytes. */
int32_t mlp_dot_c(const int8_t *a, const int8_t *b, uint32_t n);
int32_t mlp_dot_rv32(const int8_t *a, const int8_t *b, uint32_t n);
int8_t mlp_relu_q6(int32_t q12);
/* shift must be in [3,12]. */
int8_t mlp_relu_requant(int32_t accumulator, uint32_t shift);
int mlp_model_valid(const mlp_model *model);
int mlp_predict(const mlp_model *model, const int8_t input[MLP_INPUT],
                int8_t hidden[MLP_HIDDEN], int32_t scores[MLP_OUTPUT]);
int8_t mlp_pixel_q6(uint8_t pixel);
int mlp_predict_rv32(const mlp_model *model, const int8_t input[MLP_INPUT],
                     int8_t hidden[MLP_HIDDEN], int32_t scores[MLP_OUTPUT]);
#endif
