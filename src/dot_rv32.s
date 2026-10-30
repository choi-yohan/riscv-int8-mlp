    .text
    .align 2
    .globl mlp_dot_rv32
    .type mlp_dot_rv32, @function

# =======================================================
# FUNCTION: mlp_dot_rv32 - Dot product of two signed INT8 arrays (RV32IM / ILP32)
# Arguments:
#   a0 (const int8_t*) points to the first contiguous array.
#   a1 (const int8_t*) points to the second contiguous array.
#   a2 (uint32_t) is the number of elements; both strides are 1 byte.
# Returns:
#   a0 (int32_t) is the exact sum of the signed integer products.
# Preconditions:
#   - 0 <= n <= 784; each array provides at least n readable bytes.
#   - For n = 0, no array elements are read and the result is 0.
# Exceptions:
#   - No runtime argument checks or error exits are implemented.
# Registers:
#   Clobbers a0-a2 and t0-t2; no stack or callee-saved registers are used.
# =======================================================

mlp_dot_rv32:
    li t0, 0
    beqz a2, .Ldone
.Lnext:
    lb t1, 0(a0)
    lb t2, 0(a1)
    mul t1, t1, t2
    add t0, t0, t1
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, .Lnext
.Ldone:
    mv a0, t0
    ret
    .size mlp_dot_rv32, .-mlp_dot_rv32
