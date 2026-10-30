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
#   Inlines eight-element-unrolled dot products and processes two neurons per
#   input traversal in both layers. Hoists hidden ReLU rounding/cap constants,
#   requantizes both hidden values independently, and selects first output argmax.
# Exceptions:
#   - No runtime argument checks or error exits are implemented.
# Registers:
#   Uses a 64-byte stack frame and preserves the used s0-s10 registers.
#   Hidden loop: a0/a2=accumulators, a1=input cursor, a3=cap, a4=rounding offset.
#   s5=row-r cursor; row r+1 is accessed at s5+784. Clobbers t0-t6 and a0-a4.
# =======================================================

mlp_predict_rv32:
    addi sp, sp, -64
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
    mv s5, s0  # current weight row address
    li s6, MLP_B1_OFFSET  # current bias address
    add s6, s0, s6
    mv s7, s2
    li s4, 128

    # Hidden layer
    addi s8, s1, 784  # One-past-end input address, shared by all row pairs
    li a3, 127
    sll a3, a3, s10  # saturation threshold
    li a4, 1
    addi t0, s10, -1
    sll a4, a4, t0  # rounding offset
.Lhidden:
    mv a1, s1
    lw a0, 0(s6)  # accumulator for row r, initialized with b1[r]
    lw a2, 4(s6)  # accumulator for row r+1, initialized with b1[r+1]
.Lh_next:  # 8 input elements per iteration(8-unrolled), 2 output neurons simultaneously
    lb t1, 0(s5)
    lb t2, 0(a1)
    lb t3, 1(s5)
    lb t4, 1(a1)
    lb t5, 784(s5)
    lb t6, 785(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    lb t1, 2(s5)
    lb t2, 2(a1)
    lb t3, 3(s5)
    lb t4, 3(a1)
    lb t5, 786(s5)
    lb t6, 787(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    lb t1, 4(s5)
    lb t2, 4(a1)
    lb t3, 5(s5)
    lb t4, 5(a1)
    lb t5, 788(s5)
    lb t6, 789(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    lb t1, 6(s5)
    lb t2, 6(a1)
    lb t3, 7(s5)
    lb t4, 7(a1)
    lb t5, 790(s5)
    lb t6, 791(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    addi s5, s5, 8
    addi a1, a1, 8
    bne a1, s8, .Lh_next

    # Requantize row r
    blez a0, .Lh_zero0
    bge a0, a3, .Lh_clip0
    add a0, a0, a4
    srl a0, a0, s10
    j .Lh_quant1
.Lh_clip0:
    li a0, 127
    j .Lh_quant1
.Lh_zero0:
    li a0, 0
.Lh_quant1:
    # Requantize row r+1
    blez a2, .Lh_zero1
    bge a2, a3, .Lh_clip1
    add a2, a2, a4
    srl a2, a2, s10
    j .Lh_store
.Lh_clip1:
    li a2, 127
    j .Lh_store
.Lh_zero1:
    li a2, 0
.Lh_store:
    sb a0, 0(s7)
    sb a2, 1(s7)
    addi s5, s5, 784  # Skip the row to start the next pair at row r+2/r+3
    addi s6, s6, 8
    addi s7, s7, 2
    addi s4, s4, -2
    bnez s4, .Lhidden
    li s5, MLP_W2_OFFSET
    add s5, s0, s5
    li s6, MLP_B2_OFFSET
    add s6, s0, s6
    mv s7, s3
    li s4, 0
    li s8, 0
    li s9, 0x80000000

    # Output layer
    addi s10, s2, 128  # One-past-end hidden activation address
.Loutput:
    mv a1, s2
    lw a0, 0(s6)  # accumulator for output k, initialized with b2[k]
    lw a2, 4(s6)  # accumulator for output k+1, initialized with b2[k+1]
.Lo_next:  # 8 hidden elements per iteration, 2 output neurons
    lb t1, 0(s5)
    lb t2, 0(a1)
    lb t3, 1(s5)
    lb t4, 1(a1)
    lb t5, 128(s5)
    lb t6, 129(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    lb t1, 2(s5)
    lb t2, 2(a1)
    lb t3, 3(s5)
    lb t4, 3(a1)
    lb t5, 130(s5)
    lb t6, 131(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    lb t1, 4(s5)
    lb t2, 4(a1)
    lb t3, 5(s5)
    lb t4, 5(a1)
    lb t5, 132(s5)
    lb t6, 133(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    lb t1, 6(s5)
    lb t2, 6(a1)
    lb t3, 7(s5)
    lb t4, 7(a1)
    lb t5, 134(s5)
    lb t6, 135(s5)
    mul t1, t1, t2
    mul t3, t3, t4
    mul t5, t5, t2
    mul t6, t6, t4
    add a0, a0, t1
    add a0, a0, t3
    add a2, a2, t5
    add a2, a2, t6
    addi s5, s5, 8
    addi a1, a1, 8
    bne a1, s10, .Lo_next
    sw a0, 0(s7)
    sw a2, 4(s7)
    # Compare in increasing index order; retain the first maximum on ties
    ble a0, s9, .Lo_keep0
    mv s8, s4
    mv s9, a0
.Lo_keep0:
    ble a2, s9, .Lo_keep1
    addi s8, s4, 1
    mv s9, a2
.Lo_keep1:
    addi s5, s5, 128
    addi s6, s6, 8
    addi s7, s7, 8
    addi s4, s4, 2
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
    addi sp, sp, 64
    ret
    .size mlp_predict_rv32, .-mlp_predict_rv32
