#include "mlp_layout.h"
    .text
    .align 2
    .globl mlp_predict_rv32
    .type mlp_predict_rv32, @function

# =======================================================
# FUNCTION: mlp_predict_rv32 - Full INT8 784 -> 128 ReLU -> 10 forward pass
# Arguments:
#   a0 (const mlp_model*) points to the model in mlp_layout.h layout.
#   a1 (const int8_t*) points to 784 Q6 input values.
#   a2 (int8_t*) points to a writable 128-element hidden buffer.
#   a3 (int32_t*) points to a writable 10-element score buffer.
# Returns:
#   a0 (int) is the first index with the maximum output score (0-9).
#   Hidden activations and INT32 scores are written to the supplied buffers.
# Preconditions:
#   - The caller has validated the model with mlp_model_valid.
#   - Model/input/output storage is valid, appropriately aligned and disjoint.
#   - The caller provides an ILP32 ABI-aligned stack.
# Operation:
#   Uses mlp_dot_rv32 for both layers, adds biases, applies ReLU and rounded
#   saturating requantization to the hidden layer, then selects argmax.
# Exceptions:
#   - No runtime argument checks or error exits are implemented.
# Registers:
#   Uses a 64-byte stack frame and preserves ra and the used s0-s10 registers.
# =======================================================

mlp_predict_rv32:
    addi sp, sp, -64
    sw ra, 60(sp)
    sw s0, 56(sp)
    sw s1, 52(sp)
    sw s2, 48(sp)
    sw s3, 44(sp)
    sw s4, 40(sp)
    sw s5, 36(sp)
    sw s6, 32(sp)
    sw s7, 28(sp)
    sw s8, 24(sp)
    sw s9, 20(sp)
    sw s10, 16(sp)
    mv s0, a0
    mv s1, a1
    mv s2, a2
    mv s3, a3
    li t0, MLP_W1_FRAC_OFFSET
    add t0, s0, t0
    lw t1, 0(t0)
    lw t2, 4(t0)
    addi s10, t1, 6
    sub s10, s10, t2
    mv s5, s0
    li s6, MLP_B1_OFFSET
    add s6, s0, s6
    mv s7, s2
    li s4, 128
.Lhidden:
    mv a0, s1
    mv a1, s5
    li a2, 784
    call mlp_dot_rv32
    lw t0, 0(s6)
    add a0, a0, t0
    blez a0, .Lzero
    li t0, 127
    sll t0, t0, s10
    bge a0, t0, .Lclip
    li t0, 1
    addi t1, s10, -1
    sll t0, t0, t1
    add a0, a0, t0
    srl a0, a0, s10
    j .Lstore
.Lclip:
    li a0, 127
    j .Lstore
.Lzero:
    li a0, 0
.Lstore:
    sb a0, 0(s7)
    addi s5, s5, 784
    addi s6, s6, 4
    addi s7, s7, 1
    addi s4, s4, -1
    bnez s4, .Lhidden
    li s5, MLP_W2_OFFSET
    add s5, s0, s5
    li s6, MLP_B2_OFFSET
    add s6, s0, s6
    mv s7, s3
    li s4, 0
    li s8, 0
    li s9, 0x80000000
.Loutput:
    mv a0, s2
    mv a1, s5
    li a2, 128
    call mlp_dot_rv32
    lw t0, 0(s6)
    add a0, a0, t0
    sw a0, 0(s7)
    ble a0, s9, .Lkeep
    mv s8, s4
    mv s9, a0
.Lkeep:
    addi s5, s5, 128
    addi s6, s6, 4
    addi s7, s7, 4
    addi s4, s4, 1
    li t0, 10
    blt s4, t0, .Loutput
    mv a0, s8
    lw s10, 16(sp)
    lw s9, 20(sp)
    lw s8, 24(sp)
    lw s7, 28(sp)
    lw s6, 32(sp)
    lw s5, 36(sp)
    lw s4, 40(sp)
    lw s3, 44(sp)
    lw s2, 48(sp)
    lw s1, 52(sp)
    lw s0, 56(sp)
    lw ra, 60(sp)
    addi sp, sp, 64
    ret
    .size mlp_predict_rv32, .-mlp_predict_rv32
