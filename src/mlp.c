#include "mlp.h"
#include <stddef.h>
#include "mlp_layout.h"
_Static_assert(offsetof(mlp_model, w2) == MLP_W2_OFFSET, "w2 layout");
_Static_assert(offsetof(mlp_model, b1) == MLP_B1_OFFSET, "b1 layout");
_Static_assert(offsetof(mlp_model, b2) == MLP_B2_OFFSET, "b2 layout");
_Static_assert(offsetof(mlp_model, w1_frac) == MLP_W1_FRAC_OFFSET, "scales layout");
_Static_assert(sizeof(mlp_model) == MLP_MODEL_SIZE, "model layout");

/* Accumulate the dot product of two contiguous INT8 arrays into INT32. */
int32_t mlp_dot_c(const int8_t *a, const int8_t *b, uint32_t n) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < n; ++i)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

/* Round a pixel in [0,255] to a Q6 input integer in [0,64]. */
int8_t mlp_pixel_q6(uint8_t pixel) {
    return (int8_t)(((uint32_t)pixel * 64u + 127u) / 255u);
}

/* Apply ReLU to an accumulator with 12 fractional bits and convert it to Q6 INT8. */
int8_t mlp_relu_q6(int32_t q12) {
    return mlp_relu_requant(q12, 6);
}

/* Apply ReLU, rounding, a right shift, and saturation to [0,127]. */
int8_t mlp_relu_requant(int32_t acc, uint32_t shift) {
    /* shift is in [3,12] for valid models. Round half up after ReLU. */
    if (acc <= 0) return 0;
    if (acc >= (127 << shift)) return 127;
    return (int8_t)((acc + (1 << (shift - 1))) >> shift);
}


/* Validate the model pointer, fractional-bit settings, and safe bias bounds. */
int mlp_model_valid(const mlp_model *m) {
    if (!m) return 0;
    if (m->w1_frac < 4 || m->w1_frac > 10 ||
        m->w2_frac < 4 || m->w2_frac > 10 ||
        m->hidden_frac < 4 || m->hidden_frac > 7) return 0;
    for (uint32_t i = 0; i < MLP_HIDDEN; ++i)
        if (m->b1[i] < -MLP_BIAS_LIMIT || m->b1[i] > MLP_BIAS_LIMIT) return 0;
    for (uint32_t i = 0; i < MLP_OUTPUT; ++i)
        if (m->b2[i] < -MLP_BIAS_LIMIT || m->b2[i] > MLP_BIAS_LIMIT) return 0;
    return 1;
}

/* Run the 784 -> 128 ReLU -> 10 forward pass and return the first maximum score index. */
int mlp_predict(const mlp_model *m, const int8_t x[MLP_INPUT],
                int8_t h[MLP_HIDDEN], int32_t scores[MLP_OUTPUT]) {
    /* Preconditions: valid model; distinct, non-null input/output buffers.
       Maximum |dot| <= 784 * 128 * 128 = 12,845,056.
       Adding a validated bias remains safely within int32_t. */
#ifdef MLP_USE_RV32
#define DOT mlp_dot_rv32
#else
#define DOT mlp_dot_c
#endif
    for (uint32_t r = 0; r < MLP_HIDDEN; ++r)
        h[r] = mlp_relu_requant(DOT(x, m->w1 + r * MLP_INPUT, MLP_INPUT) + m->b1[r],
                              MLP_FRAC + m->w1_frac - m->hidden_frac);
    int best = 0;
    for (uint32_t r = 0; r < MLP_OUTPUT; ++r) {
        scores[r] = DOT(h, m->w2 + r * MLP_HIDDEN, MLP_HIDDEN) + m->b2[r];
        if (scores[r] > scores[best]) best = (int)r;
    }
    return best;
}
