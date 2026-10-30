# RISC-V INT8 MLP

An integer-only MNIST classifier with C and hand-written RISC-V assembly inference and on-device benchmarks for the RP2350 Hazard3 core.

The network is **784 → 128 ReLU → 10**. Weights and hidden activations use INT8 storage; dot products, biases, and output scores use INT32. The selected model uses Q2.6 weights and Q4.4 hidden activations and achieves **97.67% MNIST test accuracy**, evaluated with the host C implementation. See [QUANTIZATION.md](QUANTIZATION.md) for numerical conventions, scale selection, and training instructions.

## Assembly optimization

I started with a scalar implementation and optimized it in three stages:

- **Legacy:** a separate scalar dot-product function for each neuron.
- **Inline/unroll:** inline dot products, eight elements per loop iteration, and loads scheduled ahead of their use.
- **Optimized:** two neurons computed together to reuse input loads, with rounding and saturation constants moved outside the hidden-layer loop.

### Function inlining

Inlining the dot product removes the call/return pair and the argument setup for each neuron. A Hazard3 `jal` or `jalr` and a `ret` each take 2 cycles, so removing the pair saves 4 cycles per call. An additional `auipc` in the call sequence adds another cycle. Therefore, inlining function can reduce 4+ cycles per one call. Actual costs can be higher because of memory stalls or instruction alignment; see the [Hazard3 reference manual](https://wren.wtf/hazard3/doc/).

Once both dot products are inlined, the forward function makes no further function calls. It therefore no longer needs to save and restore `ra`. It still follows the calling convention and preserves every callee-saved register it modifies.

### Loop unrolling

I unrolled each dot-product loop by eight to reduce loop-control instructions. The 784-element hidden-layer dot product takes 98 iterations, and the 128-element output-layer dot product takes 16. Both lengths are divisible by eight, so neither loop needs remainder handling.

This removes seven out of eight loop-condition checks compared with a scalar loop. It does not remove seven out of eight arithmetic or load instructions.

### Load scheduling

On Hazard3, using a loaded value in the immediately following instruction introduces a one-cycle load-use stall under ideal bus conditions. I load operands for neighboring input elements into separate temporary registers before performing the multiplications:

```asm
lb  t1, 0(s5)  # Weight for element j
lb  t2, 0(a1)  # Input element j
lb  t3, 1(s5)  # Weight for element j+1
lb  t4, 1(a1)  # Input element j+1
mul t1, t1, t2
mul t3, t3, t4
```

The independent loads separate each load from its first use. The registers keep both operand pairs available until they are consumed; the scheduling hides the load-use latency without adding padding instructions.

### Input reuse across two neurons

For two neighboring neurons, both dot products read the same input vector. Instead of traversing that vector twice, I maintain two accumulators, initialize each with its own bias, and reuse every input value:

```text
sum0 += W[r][j]   * x[j]
sum1 += W[r+1][j] * x[j]
```

Two products now require three loads rather than four: one input and two weights. This applies to both layers. The weight rows remain contiguous in row-major storage, separated by 784 bytes in the hidden layer and 128 bytes in the output layer.

Each hidden accumulator is independently passed through ReLU, rounding, shifting, and saturation before storing its INT8 result. Output accumulators are stored directly as INT32 scores.

### Loop-invariant calculations and pointer updates

The hidden-layer shift is the same for every neuron, so I compute its rounding offset and saturation threshold once before the layer loop and keep them in registers.

Within each dot product, I advance the input and weight pointers by eight bytes per iteration. Comparing the input pointer with a precomputed end address removes the need to decrement a separate element counter. After processing a neuron pair, the code advances to the next two weight rows, biases, and output positions.

## Benchmark results

Measured on an RP2350A board, core 0 at **150 MHz**, with Pico SDK 2.3.1 and RISC-V GCC 16.1.0. All eight implementations ran in the same firmware, using one fixed MNIST image, **8 warmup runs and 16 measured runs each**, SRAM execution, masked interrupts during inference, and LTO disabled.

| Implementation | Mean cycles | Retired instructions per call | Reported CPI | Mean time (ms) |
|---|---:|---:|---:|---:|
| C `-O0` | 2,657,232 | 1,942,521 | 1.367 | 17.715 |
| C `-O1` | 868,714 | 715,337 | 1.214 | 5.791 |
| C `-O2` | 765,043 | 713,928 | 1.071 | 5.100 |
| C `-O3` | 765,043 | 713,928 | 1.071 | 5.100 |
| C `-O3 -funroll-loops` | 492,738 | 447,564 | 1.100 | 3.285 |
| ASM legacy | 944,306 | 816,275 | 1.156 | 6.295 |
| ASM inline/unroll | 491,348 | 446,619 | 1.100 | 3.276 |
| **ASM optimized** | **408,088** | **376,054** | **1.085** | **2.721** |

The optimized assembly uses **17.2% fewer cycles than C with `-O3 -funroll-loops`** (1.207× throughput) and **56.8% fewer than the legacy assembly**.

### Measurement scope

`mcycle` and `minstret` are enabled and frozen together to measure the same forward-call interval. **CPI = total cycles / total retired instructions**; a compressed instruction counts as one instruction. Raw values include argument setup, call/return, and counter-gating overhead. The measured empty-call minimum was 12 cycles and 10 instructions, which were not subtracted.

Both layers, bias additions, hidden ReLU/requantization, result stores, and argmax are included. Model loading/export, pixel conversion, correctness checks, USB output, and interrupt masking/restoration are excluded. Before timing, every implementation must match reference hidden values, scores, and predictions for four inputs.

*Note: These are repeated-input measurements, not full-dataset latency statistics. Code/data placement and instruction alignment can affect results between builds; compare implementations within the same firmware.*

## Host build and tests

Requires Python 3.10+, NumPy, a C11 compiler, and Make. Run commands from the repository root.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make test PYTHON=.venv/bin/python
./build/mlp models/recommended.i8ml models/sample.u8
```

The CLI requires an exported model and exactly 784 raw grayscale bytes in row order, with pixel values from 0 to 255. It prints the predicted digit and integer scores as JSON. The recommended model and sample image are included; MNIST data, other training artifacts, and build outputs are excluded by `.gitignore`. To retrain or compare quantization settings, follow the [training instructions](QUANTIZATION.md#reproduce-the-experiments).

## Pico build and run

Requires the Pico SDK, a Hazard3-capable RISC-V GCC toolchain, CMake, Ninja, picotool, and Python 3. Set `PICO_SDK_PATH` to the SDK directory, `PICO_TOOLCHAIN_PATH` to the toolchain root containing `bin/`, and `PICOTOOL_DIR` to the directory containing `picotoolConfig.cmake`. CMake and Ninja must be on `PATH`.

```sh
MODEL_DIR="$PWD/models"  # Use models-run for the training example.
cmake -S . -B build-pico -G Ninja \
  -DPICO_SDK_PATH="$PICO_SDK_PATH" \
  -DPICO_TOOLCHAIN_PATH="$PICO_TOOLCHAIN_PATH" \
  -Dpicotool_DIR="$PICOTOOL_DIR" \
  -DMLP_BENCH_MODEL="$MODEL_DIR/recommended.i8ml" \
  -DMLP_BENCH_IMAGE="$MODEL_DIR/sample.u8" \
  -DMLP_BENCH_REPEATS=16 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-pico -j 4
```

After source edits, repeat the build command. Hold BOOTSEL while connecting USB, copy `build-pico/mlp_benchmark.uf2` to the `RP2350` drive, then open USB serial at 115200 baud after reboot. The firmware waits for the serial connection before starting; send `r` to repeat. On macOS, find the port with `ls /dev/cu.usbmodem*`.

The default target is `pico2` / `rp2350-riscv`. CMake options `MLP_BENCH_WARMUP`, `MLP_BENCH_DISABLE_IRQ`, and `MLP_BENCH_RUN_FROM_RAM` control warmup, interrupt masking, and SRAM execution; their defaults are `8`, `ON`, and `ON`. The C presets include their helper functions; the common measurement wrapper uses `-O3`.

## QEMU validation

Requires Clang with a RISC-V backend, LLD, and `qemu-system-riscv32`, plus the model and local MNIST data.

```sh
.venv/bin/python tools/export_fixture.py \
  --data data/MNIST/raw --model models/recommended.i8ml
make rv32-check PYTHON=.venv/bin/python RVCC=clang RVLD=ld.lld
```

The Makefile checks the baseline assembly using 19 reference inputs and 28 scale-boundary combinations. QEMU checks correctness; cycle and CPI results above were measured on hardware.

## Source layout

| Path | Purpose |
|---|---|
| `src/mlp.c` | C inference and numerical primitives |
| `src/predict_rv32_{legacy,inline_unroll,optimized}.s` | Three assembly inference versions |
| `src/dot_rv32.s` | Scalar dot product used by the legacy version |
| `src/benchmark_pico.c` | Cycle, instruction-count, and CPI measurements |
| `include/` | Shared model interface and C/assembly memory layout |
| `mlp_int8/`, `tools/` | NumPy training, quantization, and data export |
| `tests/`, `rv32/` | Host tests and bare-metal QEMU validation |

Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges (standard IDX format).

License: [MIT](LICENSE).
