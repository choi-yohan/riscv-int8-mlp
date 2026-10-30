# RISC-V INT8 MLP

This project explores **RISC-V assembly optimization** through an integer-only MNIST classifier on the RP2350 Hazard3 core. It compares three assembly implementations with C compiled at several optimization levels, including `-O3 -funroll-loops`.

The ultimate goal of this project is to beat C code compiled with `-O3 -funroll-loops` in inference performance.

The network is **784 → 128 ReLU → 10**, with INT8 weights and activations and INT32 accumulation. The selected model achieves **97.67% MNIST test accuracy** with host C inference. See [QUANTIZATION.md](QUANTIZATION.md) for quantization and training details.

## Benchmark results

**The optimized assembly completes inference in 408,088 cycles (2.721 ms), using 17.2% fewer cycles than C compiled with `-O3 -funroll-loops`.**

Measured on an RP2350A board, core 0 at 150 MHz, with Pico SDK 2.3.1 and RISC-V GCC 16.1.0. All eight implementations ran in the same firmware, using one fixed MNIST image, 8 warmup runs and 16 measured runs each, SRAM execution, masked interrupts during inference, and LTO disabled.

| Implementation | Mean cycles | Instructions | CPI | Mean time (ms) |
|---|---:|---:|---:|---:|ㄴ
| C `-O0` | 2,657,232 | 1,942,521 | 1.367 | 17.715 |
| C `-O1` | 868,714 | 715,337 | 1.214 | 5.791 |
| C `-O2` | 765,043 | 713,928 | 1.071 | 5.100 |
| C `-O3` | 765,043 | 713,928 | 1.071 | 5.100 |
| C `-O3 -funroll-loops` | 492,738 | 447,564 | 1.100 | 3.285 |
| ASM legacy | 944,306 | 816,275 | 1.156 | 6.295 |
| ASM inline/unroll | 491,348 | 446,619 | 1.100 | 3.276 |
| **ASM optimized** | **408,088** | **376,054** | **1.085** | **2.721** |

These results use one fixed input. Code/data placement and instruction alignment can affect results between builds.

## Assembly optimization

The **legacy** version calls a scalar dot-product function for each neuron. The **inline/unroll** version inlines and unrolls the dot products; the **optimized** version also computes two neurons together.

- **Function inlining:** removes per-neuron call/return and argument setup. The forward function becomes a leaf function, eliminating the need to save and restore `ra`. Can reduce 4+ cycles per one function call.
- **Eight-way loop unrolling:** reduces loop-control instructions by 1/8. No remainder loop is needed.
- **Load scheduling:** places independent loads before arithmetic to avoid immediate load-use dependencies. Hazard3 requires 1 cycle stall for load-use hazard ([Hazard3 Reference Manual](https://wren.wtf/hazard3/doc/), Appendix A: Instruction Timings.)
- **Input reuse:** computes two neighboring neurons together in both layers, reusing each input value. Two products require three loads instead of four.
- **Loop simplification:** calculates rounding and saturation constants once before the hidden-layer loop and compares pointers against an end address instead of maintaining an element counter.

## Measurement method

`mcycle` and `minstret` measure the same forward-call interval. **CPI = total cycles / total retired instructions**; each compressed instruction counts as one. Results include call and counter-gating overhead; the empty-call minimum of 12 cycles and 10 instructions is not subtracted.

Timing covers both layers, bias additions, hidden ReLU/requantization, result stores, and argmax. Model loading, pixel conversion, correctness checks, USB output, and interrupt masking/restoration are excluded. Before timing, all eight implementations must match reference hidden values, scores, and predictions for four inputs.

## Host build and tests

Requires Python 3.10+, NumPy, a C11 compiler, and Make. Run commands from the repository root.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make test PYTHON=.venv/bin/python
./build/mlp models/recommended.i8ml models/sample.u8
```

The included sample contains 784 raw grayscale bytes (0–255), stored in row order. The CLI prints the predicted digit and integer scores as JSON. See the [training instructions](QUANTIZATION.md#reproduce-the-experiments) to retrain the model.

## Pico build and run

Requires the Pico SDK, a Hazard3-capable RISC-V GCC toolchain, CMake, Ninja, picotool, and Python 3. Set `PICO_SDK_PATH` to the SDK directory, `PICO_TOOLCHAIN_PATH` to the toolchain root containing `bin/`, and `PICOTOOL_DIR` to the directory containing `picotoolConfig.cmake`. CMake and Ninja must be on `PATH`.

```sh
cmake -S . -B build-pico -G Ninja \
  -DPICO_SDK_PATH="$PICO_SDK_PATH" \
  -DPICO_TOOLCHAIN_PATH="$PICO_TOOLCHAIN_PATH" \
  -Dpicotool_DIR="$PICOTOOL_DIR" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-pico -j 4
```

The default configuration uses the included model and sample, targets `pico2` / `rp2350-riscv`, and measures 16 runs after 8 warmups. To rebuild after edits, repeat the build command.

Hold BOOTSEL while connecting USB, copy `build-pico/mlp_benchmark.uf2` to the `RP2350` drive, then open USB serial at 115200 baud. The benchmark starts when serial connects; send `r` to repeat. On macOS, find the port with `ls /dev/cu.usbmodem*`.

## QEMU validation

Requires Clang with a RISC-V backend, LLD, and `qemu-system-riscv32`, plus the model and local MNIST data.

```sh
.venv/bin/python tools/export_fixture.py \
  --data data/MNIST/raw --model models/recommended.i8ml
make rv32-check PYTHON=.venv/bin/python RVCC=clang RVLD=ld.lld
```

The Makefile checks the baseline assembly using 19 reference inputs and 28 scale-boundary combinations. QEMU checks correctness; cycle and CPI results above were measured on hardware.

AI tools were used for language refinement.

Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges (standard IDX format).

License: [MIT](LICENSE).
