# Quantization

The model uses power-of-two, symmetric INT8 quantization with zero point 0. Each layer has one weight scale. Biases and output scores remain INT32.

The selected configuration is **Q2.6 weights / Q4.4 hidden activations / Q2.6 weights**. Input pixels always use six fractional bits.

## Representation and arithmetic

A stored integer with `f` fractional bits represents `integer / 2^f`. The Q notation below counts the sign bit as part of the integer bits.

| Format | Step size | Signed INT8 range |
|---|---:|---:|
| Q2.6 | 1/64 | −2 to 1.984375 |
| Q3.5 | 1/32 | −4 to 3.96875 |
| Q4.4 | 1/16 | −8 to 7.9375 |

Pixels are converted to `round(pixel * 64 / 255)`, producing integers from 0 to 64. Each weight row belongs to one output neuron; both weight matrices are stored in row-major order.

```text
acc[j] = sum_i(x[i] * w1[j,i]) + b1[j]
shift  = 6 + w1_frac - hidden_frac
h[j]   = clip((max(acc[j], 0) + 2^(shift-1)) >> shift, 0, 127)
s[k]   = sum_j(h[j] * w2[k,j]) + b2[k]
class  = first_argmax(s)
```

Products accumulate without intermediate truncation. Hidden outputs use round-half-up after ReLU and saturate at 127. There is no output-layer ReLU, softmax, or INT8 requantization: all ten scores share a scale and can be compared directly.

Bias `b1` has `6 + w1_frac` fractional bits; `b2` and the output scores have `hidden_frac + w2_frac` fractional bits. For the selected model, these are 12 and 10, and the hidden-layer shift is 8. Weight and bias export rounds to the nearest integer, with ties away from zero.

The supported weight fractional-bit counts are 4–10, and the hidden count is 4–7. Bias magnitude is limited to `2^24`. The maximum absolute sum of 784 signed-byte products is 12,845,056, so validated models fit safely in INT32. Callers must supply a valid model and non-overlapping input/output buffers.

## Accuracy and range selection

Training used NumPy, seed 42, Adam, 10 epochs, batch size 256, an initial learning rate of 0.001, and a weight regularization coefficient of `1e-4`. The 60,000 training images were split into 55,000 training and 5,000 validation examples. The checkpoint with the best validation accuracy was retained. The separate 10,000-image test set was not used for training or scale selection.

| W1 / hidden / W2 | Validation accuracy | Test accuracy | Test hidden values equal to 127 |
|---|---:|---:|---:|
| FP32 | 97.34% | 97.63% | — |
| Q2.6 / Q2.6 / Q2.6 | 96.76% | 96.65% | 17.4341% |
| Q3.5 / Q3.5 / Q3.5 | 97.24% | 97.48% | 1.4768% |
| Q2.6 / Q3.5 / Q2.6 | 97.28% | 97.55% | 1.4494% |
| **Q2.6 / Q4.4 / Q2.6** | **97.36%** | **97.67%** | **0.0100%** |

All INT8 rows use the same Q6 input encoding. The last column counts `h == 127`, including values rounded to 127; it is not strictly a clipping rate.

The selected scales came from 27 combinations: W1 fractional bits `{6,7,8}`, hidden `{4,5,6}`, and W2 `{5,6,7}`. Candidates were ranked by validation accuracy, then lower hidden saturation, then lower combined weight RMSE. The Q3.5 configurations were evaluated separately on the same trained checkpoint.

### Weight distribution

| Layer | Minimum | Maximum | 99th percentile of absolute value | 99.9th percentile |
|---|---:|---:|---:|---:|
| W1, 128 × 784 | −0.489864 | 0.356212 | 0.191094 | 0.290050 |
| W2, 10 × 128 | −0.868269 | 0.766777 | 0.638903 | 0.847317 |

Neither Q2.6 nor Q3.5 clips these weights. Q2.6 weight RMSE is approximately 0.004061 for W1 and 0.004497 for W2, compared with 0.007935 and 0.008896 for Q3.5. Keeping the finer weight scale while widening the hidden range retains weight precision and reduces hidden saturation.

These results use one training seed. The selected model exceeds FP32 test accuracy by four images; this does not establish a general accuracy advantage for quantization. Full test-set accuracy was evaluated on the host, not on the board.

## Reproduce the experiments

Place the standard MNIST IDX files, decompressed or gzip-compressed, in `data/MNIST/raw/`. The tools read local files and do not download the dataset. Set up the Python environment and build the host library as described in [README.md](README.md#host-build-and-tests), then run:

```sh
.venv/bin/python tools/train.py \
  --data data/MNIST/raw --out models-run --epochs 10 --seed 42
.venv/bin/python tools/compare_quantization.py \
  --data data/MNIST/raw --models models-run
.venv/bin/python tools/compare_q3_5.py \
  --data data/MNIST/raw --models models-run
./build/mlp models-run/recommended.i8ml models-run/sample.u8
```

Use a fresh output directory for each run. Training and comparison tools reject existing model/report outputs. Numerical results may vary across NumPy and BLAS environments.

The output directory contains the FP32 checkpoint, validation indices, quantized `.i8ml` models, a raw sample image, and JSON reports. `recommended.i8ml` is selected by the validation search; `uniform_q6.i8ml`, `uniform_q3_5.i8ml`, and `weights_q2_6_hidden_q3_5.i8ml` provide comparison configurations.

## I8ML file format

The custom `.i8ml` format is little-endian and contains **102,220 bytes** for this architecture.

| Order | Contents |
|---|---|
| 4 bytes | ASCII magic `I8ML` |
| 8 × uint32 | Version 1, input size 784, hidden size 128, output size 10, input fractional bits 6, W1 fractional bits, hidden fractional bits, W2 fractional bits |
| 100,352 × int8 | W1, row-major |
| 1,280 × int8 | W2, row-major |
| 128 × int32 | b1 |
| 10 × int32 | b2 |

The format stores the fractional-bit counts; inference reads these values from the model. C model field offsets are checked at compile time against the assembly layout. Model loading validates the header, supported scales, bias bounds, and exact file length.

Quantization experiments and analysis were performed with assistance from OpenAI Codex (GPT-6.1 Sol).