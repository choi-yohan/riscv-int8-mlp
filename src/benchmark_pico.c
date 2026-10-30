/* Pico 2 / RP2350 Hazard3 benchmark. Build with the root CMakeLists.txt.
 * USB printf, correctness checks, and IRQ masking/restoration are NOT timed.
 * Compile mlp.c without MLP_USE_RV32 so the C baseline really uses C dot products.
 */
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/clocks.h"
#include "hardware/riscv.h"
#include "hardware/sync.h"
#include "mlp.h"
#include "benchmark_data.h"

#if !defined(__riscv) || __riscv_xlen != 32
#error "Build for PICO_PLATFORM=rp2350-riscv"
#endif
#ifdef MLP_USE_RV32
#error "The benchmark C baseline must be built without MLP_USE_RV32"
#endif

#ifndef MLP_BENCH_REPEATS
#define MLP_BENCH_REPEATS 16
#endif
#ifndef MLP_BENCH_WARMUP
#define MLP_BENCH_WARMUP 8
#endif
#ifndef MLP_BENCH_DISABLE_IRQ
#define MLP_BENCH_DISABLE_IRQ 1
#endif
#ifndef MLP_BENCH_RUN_FROM_RAM
#define MLP_BENCH_RUN_FROM_RAM 1
#endif
_Static_assert(MLP_BENCH_REPEATS > 0 && MLP_BENCH_REPEATS <= 1000000,
               "repeat count must be in [1,1000000]");
_Static_assert(MLP_BENCH_WARMUP >= 0, "warmup count must be nonnegative");

/* The CSR immediate 5 selects CY (bit 0) and IR (bit 2) together. */
_Static_assert((RVCSR_MCOUNTINHIBIT_CY_BITS | RVCSR_MCOUNTINHIBIT_IR_BITS) == 5,
               "cycle and instruction counter gate mask");

typedef int (*predict_fn)(const mlp_model *, const int8_t *, int8_t *, int32_t *);
/* Each entry point and its C helpers are compiled at the named optimization level. */
#define DECLARE_C_VARIANT(level) \
    int mlp_predict_o##level(const mlp_model *, const int8_t *, int8_t *, int32_t *)
DECLARE_C_VARIANT(0);
DECLARE_C_VARIANT(1);
DECLARE_C_VARIANT(2);
DECLARE_C_VARIANT(3);
DECLARE_C_VARIANT(3_unroll);
#undef DECLARE_C_VARIANT

#define DECLARE_ASM_VARIANT(name) \
    int mlp_predict_rv32_##name(const mlp_model *, const int8_t *, int8_t *, int32_t *)
DECLARE_ASM_VARIANT(legacy);
DECLARE_ASM_VARIANT(inline_unroll);
DECLARE_ASM_VARIANT(optimized);
#undef DECLARE_ASM_VARIANT

typedef struct { const char *name; predict_fn predict; } implementation;
static const implementation implementations[] = {
    {"C -O0", mlp_predict_o0},
    {"C -O1", mlp_predict_o1},
    {"C -O2", mlp_predict_o2},
    {"C -O3", mlp_predict_o3},
    {"C -O3+unroll", mlp_predict_o3_unroll},
    {"RV32 legacy", mlp_predict_rv32_legacy},
    {"RV32 inline_unroll", mlp_predict_rv32_inline_unroll},
    {"RV32 optimized", mlp_predict_rv32_optimized},
};
enum { IMPLEMENTATION_COUNT = sizeof(implementations) / sizeof(implementations[0]),
       RV32_INDEX = IMPLEMENTATION_COUNT - 1 };

typedef struct { uint64_t cycles, instructions; } measurement;
typedef struct {
    uint64_t minimum, maximum, sum, instruction_sum;
    unsigned count;
} cycle_stats;
typedef struct { int8_t hidden[MLP_HIDDEN]; int32_t scores[MLP_OUTPUT]; int label; } output;

/* RV32: retry if the low half wrapped while reading the high/low halves.
 * Each asm is also a compiler memory barrier, keeping the inference between reads.
 * Hazard3 is in-order. These are CPU cycles, not SDK microsecond timer ticks.
 */
/* Read the 64-bit CPU cycle counter, retrying if the upper half changes. */
static inline uint64_t read_cycles(void) {
    uint32_t hi, lo, again;
    do {
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi) :: "memory");
        __asm__ volatile ("csrr %0, mcycle" : "=r"(lo) :: "memory");
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(again) :: "memory");
    } while (hi != again);
    return ((uint64_t)hi << 32) | lo;
}

/* Read the 64-bit retired-instruction counter, retrying across rollover. */
static inline uint64_t read_instructions(void) {
    uint32_t hi, lo, again;
    do {
        __asm__ volatile ("csrr %0, minstreth" : "=r"(hi) :: "memory");
        __asm__ volatile ("csrr %0, minstret" : "=r"(lo) :: "memory");
        __asm__ volatile ("csrr %0, minstreth" : "=r"(again) :: "memory");
    } while (hi != again);
    return ((uint64_t)hi << 32) | lo;
}

/* Accumulate cycle and retired-instruction measurements for one implementation. */
static void add_sample(cycle_stats *s, measurement sample) {
    if (!s->count || sample.cycles < s->minimum) s->minimum = sample.cycles;
    if (!s->count || sample.cycles > s->maximum) s->maximum = sample.cycles;
    s->sum += sample.cycles;
    s->instruction_sum += sample.instructions;
    ++s->count;
}

/* Check the predicted class, hidden activations, and scores against the selected golden case. */
static bool matches(const output *out, unsigned sample) {
    if (out->label != benchmark_predictions[sample]) return false;
    for (unsigned i = 0; i < MLP_HIDDEN; ++i)
        if (out->hidden[i] != benchmark_hidden[sample][i]) return false;
    for (unsigned i = 0; i < MLP_OUTPUT; ++i)
        if (out->scores[i] != benchmark_scores[sample][i]) return false;
    return true;
}

/* Compare every C and assembly variant against all golden input cases. */
static bool check_correctness(void) {
    output out;
    for (unsigned sample = 0; sample < BENCHMARK_CASES; ++sample) {
        for (unsigned impl = 0; impl < IMPLEMENTATION_COUNT; ++impl) {
            out.label = implementations[impl].predict(&benchmark_model,
                benchmark_inputs[sample], out.hidden, out.scores);
            if (!matches(&out, sample)) {
                printf("FAIL: golden mismatch, sample=%u impl=%s label=%d expected=%d\n",
                       sample, implementations[impl].name, out.label, benchmark_predictions[sample]);
                return false;
            }
            sleep_ms(1);
        }
    }
    printf("PASS: %u implementations x %u golden cases, hidden + scores + argmax\n",
           (unsigned)IMPLEMENTATION_COUNT, BENCHMARK_CASES);
    return true;
}

/* All implementations use the same wrapper, with LTO and cloning disabled.
 * CY and IR are frozen together for snapshots, then run over one shared interval.
 * Raw results include argument setup, the indirect call/return, and counter-gating
 * overhead. Snapshot reads and IRQ save/restore are outside that interval.
 */
/* Measure cycles and retired instructions over the same gated forward-call interval. */
static __attribute__((noinline, noclone)) measurement measure(predict_fn fn, output *out) {
#if MLP_BENCH_DISABLE_IRQ
    uint32_t irq_state = save_and_disable_interrupts();
#endif
    uint32_t inhibit_before;
    __asm__ volatile ("csrrsi %0, mcountinhibit, 5" : "=r"(inhibit_before) :: "memory");
    uint64_t begin_cycles = read_cycles();
    uint64_t begin_instructions = read_instructions();
    __asm__ volatile ("csrci mcountinhibit, 5" ::: "memory");
    int label = fn(&benchmark_model, benchmark_inputs[0], out->hidden, out->scores);
    __asm__ volatile ("csrsi mcountinhibit, 5" ::: "memory");
    uint64_t end_cycles = read_cycles();
    uint64_t end_instructions = read_instructions();
    __asm__ volatile ("csrw mcountinhibit, %0" :: "r"(inhibit_before) : "memory");
#if MLP_BENCH_DISABLE_IRQ
    restore_interrupts(irq_state);
#endif
    out->label = label;
    return (measurement){end_cycles - begin_cycles, end_instructions - begin_instructions};
}

/* Return without inference to expose the shared wrapper's measurement overhead. */
static __attribute__((noinline, noclone)) int empty_predict(const mlp_model *model,
        const int8_t *input, int8_t *hidden, int32_t *scores) {
    (void)model; (void)input; (void)hidden; (void)scores;
    return 0;
}

/* Find the lowest-cycle empty-call sample without subtracting it from results. */
static measurement counter_overhead(void) {
    measurement best = {UINT64_MAX, 0};
    output out;
    for (unsigned i = 0; i < 64; ++i) {
        measurement sample = measure(empty_predict, &out);
        if (sample.cycles < best.cycles) best = sample;
    }
    return best;
}

/* Print raw cycles, mean time, mean retired instructions, and aggregate CPI. */
static void print_stats(const char *name, const cycle_stats *s, uint32_t hz) {
    /* Fixed-point output avoids pulling floating-point printf into firmware. */
    uint64_t mean_cycles_x100 = (s->sum / s->count) * 100u +
                               (s->sum % s->count) * 100u / s->count;
    uint64_t mean_us_x100 = (mean_cycles_x100 / hz) * UINT64_C(1000000) +
                           (mean_cycles_x100 % hz) * UINT64_C(1000000) / hz;
    uint64_t instructions_x100 = (s->instruction_sum / s->count) * 100u +
                                 (s->instruction_sum % s->count) * 100u / s->count;
    uint64_t cpi_x1000 = s->instruction_sum ?
        (s->sum / s->instruction_sum) * 1000u +
        (s->sum % s->instruction_sum) * 1000u / s->instruction_sum : 0;
    printf("%s: n=%u min=%" PRIu64 " mean=%" PRIu64 ".%02" PRIu64
           " max=%" PRIu64 " cycles; mean=%" PRIu64 ".%02" PRIu64 " us; "
           "instructions=%" PRIu64 ".%02" PRIu64 "; CPI=%" PRIu64 ".%03" PRIu64 "\n",
           name, s->count, s->minimum, mean_cycles_x100 / 100, mean_cycles_x100 % 100,
           s->maximum, mean_us_x100 / 100, mean_us_x100 % 100,
           instructions_x100 / 100, instructions_x100 % 100,
           cpi_x1000 / 1000, cpi_x1000 % 1000);
}

/* Validate and compare five C presets and three assembly variants using cycles and CPI. */
static bool benchmark(void) {
    cycle_stats stats[IMPLEMENTATION_COUNT] = {0};
    output out;
    const uint32_t hz = clock_get_hz(clk_sys);
    printf("\nINT8 784-128-10 | RP2350 RISC-V | core=%u | clk_sys=%" PRIu32 " Hz\n",
           get_core_num(), hz);
    printf("build: C -O0/-O1/-O2/-O3/-O3+funroll-loops; ASM legacy/inline_unroll/optimized; harness -O3; LTO off; run_from_ram=%d; IRQs_masked=%d\n",
           MLP_BENCH_RUN_FROM_RAM, MLP_BENCH_DISABLE_IRQ);
    printf("model: w1_frac=%" PRIu32 " hidden_frac=%" PRIu32 " w2_frac=%" PRIu32 "\n",
           benchmark_model.w1_frac, benchmark_model.hidden_frac, benchmark_model.w2_frac);
    printf("model SHA256: %s\ninput SHA256: %s\n", BENCHMARK_MODEL_SHA256, BENCHMARK_IMAGE_SHA256);
    if (!mlp_model_valid(&benchmark_model) || !check_correctness()) return false;

    /* Hazard3 counters reset inhibited; enable both before probing them. */
    riscv_clear_csr(mcountinhibit, RVCSR_MCOUNTINHIBIT_CY_BITS | RVCSR_MCOUNTINHIBIT_IR_BITS);
    uint64_t probe = read_cycles();
    uint64_t instruction_probe = read_instructions();
    busy_wait_us_32(10);
    if (read_cycles() == probe) {
        puts("FAIL: mcycle is not advancing");
        return false;
    }
    if (read_instructions() == instruction_probe) {
        puts("FAIL: minstret is not advancing");
        return false;
    }
    measurement overhead = counter_overhead();
    printf("empty-call minimum: %" PRIu64 " cycles, %" PRIu64
           " instructions (not subtracted)\n", overhead.cycles, overhead.instructions);
    puts("CPI = total raw cycles / total retired instructions; shared gated interval");

    for (int i = 0; i < MLP_BENCH_WARMUP; ++i) {
        for (unsigned impl = 0; impl < IMPLEMENTATION_COUNT; ++impl) {
            (void)measure(implementations[impl].predict, &out);
            if (!matches(&out, 0)) {
                printf("FAIL: warmup output changed, impl=%s\n", implementations[impl].name);
                return false;
            }
            sleep_ms(1);
        }
    }
    printf("warming complete; %d measured inferences per implementation, fixed image #0\n",
           MLP_BENCH_REPEATS);
    stdio_flush();
    sleep_ms(20);
    for (unsigned i = 0; i < MLP_BENCH_REPEATS; ++i) {
        /* Rotate the starting implementation to reduce execution-order effects. */
        for (unsigned step = 0; step < IMPLEMENTATION_COUNT; ++step) {
            unsigned impl = (i + step) % IMPLEMENTATION_COUNT;
            measurement sample = measure(implementations[impl].predict, &out);
            if (!sample.instructions) {
                puts("FAIL: no instructions retired in measurement window");
                return false;
            }
            add_sample(&stats[impl], sample);
            if (!matches(&out, 0)) {
                printf("FAIL: output changed, repetition=%u impl=%s\n",
                       i, implementations[impl].name);
                return false;
            }
            sleep_ms(1);
        }
    }
    for (unsigned impl = 0; impl < IMPLEMENTATION_COUNT; ++impl)
        print_stats(implementations[impl].name, &stats[impl], hz);
    for (unsigned impl = 0; impl < RV32_INDEX; ++impl) {
        uint64_t ratio_x1000 = stats[impl].sum * 1000u / stats[RV32_INDEX].sum;
        printf("%s/RV32 optimized mean cycle ratio: %" PRIu64 ".%03" PRIu64 "x\n",
               implementations[impl].name, ratio_x1000 / 1000, ratio_x1000 % 1000);
    }
    puts("PASS: benchmark complete. Send 'r' to repeat; BOOTSEL to upload another UF2.");
    return true;
}

/* Wait for USB serial, run the benchmark, and rerun when r or R is received. */
int main(void) {
    stdio_init_all();
    /* Do not lose a one-shot report while the host is still opening its terminal. */
    while (!stdio_usb_connected()) sleep_ms(10);
    sleep_ms(100);
    (void)benchmark();
    while (true) {
        int ch = getchar_timeout_us(10000);
        if (ch == 'r' || ch == 'R') (void)benchmark();
    }
}
