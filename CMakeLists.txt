cmake_minimum_required(VERSION 3.20)

# Pico SDK build only. Existing host/QEMU Makefile targets remain separate.
set(PICO_BOARD pico2 CACHE STRING "Pico-compatible RP2350 board")
set(PICO_PLATFORM rp2350-riscv CACHE STRING "RISC-V is required for mcycle")
if(NOT PICO_PLATFORM STREQUAL "rp2350-riscv")
    message(FATAL_ERROR "Use -DPICO_PLATFORM=rp2350-riscv in a fresh build directory")
endif()
if(NOT DEFINED PICO_SDK_PATH AND DEFINED ENV{PICO_SDK_PATH})
    set(PICO_SDK_PATH "$ENV{PICO_SDK_PATH}")
endif()
if(NOT DEFINED PICO_SDK_PATH)
    message(FATAL_ERROR "Set PICO_SDK_PATH to your installed Pico SDK directory")
endif()
include("${PICO_SDK_PATH}/pico_sdk_init.cmake")
project(mlp_benchmark C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
pico_sdk_init()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(MLP_BENCH_MODEL "${CMAKE_CURRENT_LIST_DIR}/models/recommended.i8ml" CACHE FILEPATH "Trained I8ML model")
set(MLP_BENCH_IMAGE "${CMAKE_CURRENT_LIST_DIR}/models/sample.u8" CACHE FILEPATH "784-byte raw image")
set(MLP_BENCH_REPEATS 16 CACHE STRING "Measured inferences per implementation")
set(MLP_BENCH_WARMUP 8 CACHE STRING "Warmup inferences per implementation")
option(MLP_BENCH_DISABLE_IRQ "Mask interrupts during each measured inference" ON)
option(MLP_BENCH_RUN_FROM_RAM "Copy firmware and constant model to SRAM at boot" ON)
foreach(asset IN ITEMS MLP_BENCH_MODEL MLP_BENCH_IMAGE)
    if(NOT EXISTS "${${asset}}")
        message(FATAL_ERROR "${asset} is missing: ${${asset}}")
    endif()
endforeach()

set(bench_header "${CMAKE_CURRENT_BINARY_DIR}/benchmark_data.h")
add_custom_command(OUTPUT "${bench_header}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tools/export_benchmark_data.py"
            --model "${MLP_BENCH_MODEL}" --image "${MLP_BENCH_IMAGE}" --out "${bench_header}"
    DEPENDS tools/export_benchmark_data.py "${MLP_BENCH_MODEL}" "${MLP_BENCH_IMAGE}"
    VERBATIM)
add_executable(mlp_benchmark src/benchmark_pico.c src/mlp.c
               src/dot_rv32.s "${bench_header}")
# Compile baseline levels and an explicit O3/unroll preset with their own helpers.
# Rename every public C symbol so each variant calls helpers at its own level.
foreach(preset IN ITEMS o0 o1 o2 o3 o3_unroll)
    set(variant mlp_c_${preset})
    if(preset STREQUAL "o3_unroll")
        set(optimization_flags -O3 -funroll-loops)
    else()
        string(SUBSTRING "${preset}" 1 1 level)
        set(optimization_flags -O${level})
    endif()
    add_library(${variant} OBJECT src/mlp.c)
    target_include_directories(${variant} PRIVATE include)
    target_compile_options(${variant} PRIVATE ${optimization_flags} -Wall -Wextra -Werror -fno-lto)
    set_property(TARGET ${variant} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    foreach(symbol IN ITEMS mlp_dot_c mlp_pixel_q6 mlp_relu_q6
                            mlp_relu_requant mlp_model_valid mlp_predict)
        target_compile_definitions(${variant} PRIVATE ${symbol}=${symbol}_${preset})
    endforeach()
    target_sources(mlp_benchmark PRIVATE $<TARGET_OBJECTS:${variant}>)
endforeach()
# Give each assembly implementation a distinct exported entry point.
foreach(preset IN ITEMS legacy inline_unroll optimized)
    set(variant mlp_asm_${preset})
    add_library(${variant} OBJECT src/predict_rv32_${preset}.s)
    target_include_directories(${variant} PRIVATE include)
    target_compile_definitions(${variant} PRIVATE mlp_predict_rv32=mlp_predict_rv32_${preset})
    target_compile_options(${variant} PRIVATE -x assembler-with-cpp -fno-lto)
    set_property(TARGET ${variant} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_sources(mlp_benchmark PRIVATE $<TARGET_OBJECTS:${variant}>)
endforeach()
target_include_directories(mlp_benchmark PRIVATE include "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(mlp_benchmark pico_stdlib hardware_clocks hardware_sync hardware_riscv)
target_compile_options(mlp_benchmark PRIVATE -O3 -Wall -Wextra -Werror -fno-lto)
# Lowercase .s still needs preprocessing for the shared layout header.
target_compile_options(mlp_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:ASM>:-x;assembler-with-cpp>")
target_link_options(mlp_benchmark PRIVATE -fno-lto)
set_property(TARGET mlp_benchmark PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_definitions(mlp_benchmark PRIVATE
    MLP_BENCH_REPEATS=${MLP_BENCH_REPEATS}
    MLP_BENCH_WARMUP=${MLP_BENCH_WARMUP}
    MLP_BENCH_DISABLE_IRQ=$<BOOL:${MLP_BENCH_DISABLE_IRQ}>
    MLP_BENCH_RUN_FROM_RAM=$<BOOL:${MLP_BENCH_RUN_FROM_RAM}>)
if(MLP_BENCH_RUN_FROM_RAM)
    pico_set_binary_type(mlp_benchmark copy_to_ram)
endif()
pico_enable_stdio_usb(mlp_benchmark 1)
pico_enable_stdio_uart(mlp_benchmark 0)
pico_set_program_name(mlp_benchmark "INT8 MLP cycle benchmark")
pico_add_extra_outputs(mlp_benchmark)
