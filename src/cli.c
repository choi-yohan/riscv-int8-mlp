#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read one little-endian unsigned 32-bit integer from a file. */
static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

/* Load an I8ML model and validate its format and numeric bounds. */
static int load_model(const char *path, mlp_model *m) {
    FILE *f = fopen(path, "rb");
    char magic[4];
    uint32_t field;
    const uint32_t expected[] = {1, MLP_INPUT, MLP_HIDDEN, MLP_OUTPUT, MLP_FRAC};
    if (!f) return 0;
    int ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "I8ML", 4) == 0;
    for (unsigned i = 0; ok && i < 5; ++i)
        ok = read_u32(f, &field) && field == expected[i];
    if (ok) ok = read_u32(f, &m->w1_frac);
    if (ok) ok = read_u32(f, &m->hidden_frac);
    if (ok) ok = read_u32(f, &m->w2_frac);
    if (ok) ok = fread(m->w1, 1, sizeof(m->w1), f) == sizeof(m->w1);
    if (ok) ok = fread(m->w2, 1, sizeof(m->w2), f) == sizeof(m->w2);
    for (unsigned i = 0; ok && i < MLP_HIDDEN + MLP_OUTPUT; ++i) {
        ok = read_u32(f, &field);
        if (ok) {
            int64_t value = field <= INT32_MAX ? (int64_t)field : (int64_t)field - 4294967296LL;
            if (i < MLP_HIDDEN) m->b1[i] = (int32_t)value;
            else m->b2[i - MLP_HIDDEN] = (int32_t)value;
        }
    }
    ok = ok && fgetc(f) == EOF && !ferror(f) && mlp_model_valid(m);
    if (fclose(f) != 0) ok = 0;
    return ok;
}

/* Load a model and raw image, run inference, and print the prediction and integer scores as JSON. */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.i8ml image.u8 (784 raw grayscale bytes)\n", argv[0]);
        return 2;
    }
    mlp_model *m = malloc(sizeof(*m));
    if (!m || !load_model(argv[1], m)) {
        fprintf(stderr, "invalid model or unreadable file\n"); free(m); return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    uint8_t pixels[MLP_INPUT];
    if (!f) { free(m); return 2; }
    int ok = fread(pixels, 1, sizeof(pixels), f) == sizeof(pixels);
    ok = ok && fgetc(f) == EOF && !ferror(f);
    if (fclose(f) != 0) ok = 0;
    if (!ok) { fprintf(stderr, "image must contain exactly 784 bytes\n"); free(m); return 2; }
    int8_t x[MLP_INPUT], h[MLP_HIDDEN];
    int32_t scores[MLP_OUTPUT];
    for (unsigned i = 0; i < MLP_INPUT; ++i) x[i] = mlp_pixel_q6(pixels[i]);
    int label = mlp_predict(m, x, h, scores);
    printf("{\"prediction\":%d,\"scores_frac\":%u,\"scores\":[", label, m->hidden_frac + m->w2_frac);
    for (int i = 0; i < MLP_OUTPUT; ++i) printf("%s%ld", i ? "," : "", (long)scores[i]);
    puts("]}");
    free(m);
    return 0;
}
