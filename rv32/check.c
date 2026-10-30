#include "mlp.h"
#include "fixture.h"

static mlp_model boundary_model;

/* Check ReLU, rounding, saturation, and tie handling across 28 scale combinations. */
static int check_scale_boundaries(void) {
    int8_t input[MLP_INPUT] = {0}, hc[MLP_HIDDEN], ha[MLP_HIDDEN];
    int32_t sc[MLP_OUTPUT], sa[MLP_OUTPUT];
    boundary_model.w2_frac = 6;
    for (unsigned f1 = 4; f1 <= 10; ++f1) {
        for (unsigned fh = 4; fh <= 7; ++fh) {
            boundary_model.w1_frac = f1;
            boundary_model.hidden_frac = fh;
            unsigned shift = 6 + f1 - fh;
            int32_t half = 1 << (shift - 1);
            int32_t cap = 127 << shift;
            const int32_t values[] = {-MLP_BIAS_LIMIT, -1, 0, half - 1, half,
                                      cap - half - 1, cap - half, cap, MLP_BIAS_LIMIT};
            for (int i = 0; i < MLP_HIDDEN; ++i)
                boundary_model.b1[i] = values[i % 9];
            /* Nonzero output weights exercise activation propagation and ties. */
            for (int i = 0; i < MLP_OUTPUT * MLP_HIDDEN; ++i)
                boundary_model.w2[i] = (int8_t)(i % 255 - 128);
            if (!mlp_model_valid(&boundary_model)) return 9;
            int lc = mlp_predict(&boundary_model, input, hc, sc);
            int la = mlp_predict_rv32(&boundary_model, input, ha, sa);
            if (lc != la) return 10;
            for (int j = 0; j < MLP_HIDDEN; ++j)
                if (hc[j] != ha[j]) return 11;
            for (int j = 0; j < MLP_OUTPUT; ++j)
                if (sc[j] != sa[j]) return 12;
        }
    }
    /* Zero model gives equal scores; both implementations must choose class 0. */
    for (int i = 0; i < MLP_OUTPUT * MLP_HIDDEN; ++i) boundary_model.w2[i] = 0;
    return mlp_predict_rv32(&boundary_model, input, ha, sa) == 0 ? 0 : 13;
}

/* Validate dot products and forward passes, returning zero on success or a check-specific error code. */
int check_main(void) {
    int8_t a[784], b[784], hidden[MLP_HIDDEN];
    int32_t scores[MLP_OUTPUT];
    for (int i = 0; i < 784; ++i) {
        a[i] = (int8_t)(i % 256 - 128);
        b[i] = (int8_t)((i * 37) % 256 - 128);
    }
    const uint32_t lengths[] = {0, 1, 3, 127, 128, 783, 784};
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i)
        if (mlp_dot_c(a, b, lengths[i]) != mlp_dot_rv32(a, b, lengths[i])) return 1;
    if (!mlp_model_valid(&fixture_model)) return 2;
    for (int n = 0; n < FIXTURE_COUNT; ++n) {
        int label = mlp_predict(&fixture_model, fixture_inputs[n], hidden, scores);
        if (label != fixture_labels[n]) return 3;
        for (int j = 0; j < MLP_OUTPUT; ++j)
            if (scores[j] != fixture_scores[n][j]) return 4;
        for (int j = 0; j < MLP_HIDDEN; ++j)
            if (hidden[j] != fixture_hidden[n][j]) return 5;
        label = mlp_predict_rv32(&fixture_model, fixture_inputs[n], hidden, scores);
        if (label != fixture_labels[n]) return 6;
        for (int j = 0; j < MLP_OUTPUT; ++j)
            if (scores[j] != fixture_scores[n][j]) return 7;
        for (int j = 0; j < MLP_HIDDEN; ++j)
            if (hidden[j] != fixture_hidden[n][j]) return 8;
    }
    return check_scale_boundaries();
}
