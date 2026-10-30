# =======================================================
# FUNCTION: _start - Bare-metal entry point for the QEMU virt checks
# Arguments:
#   None; entered by the emulator rather than through a C function call.
# Returns:
#   Does not return; writes the check result to the QEMU virt test device.
# Preconditions:
#   - rv32/link.ld supplies __stack_top, __bss_start and __bss_end.
#   - The QEMU virt test device is mapped at 0x100000.
# Operation:
#   Initializes sp, clears BSS, calls check_main, and signals pass (0x5555)
#   or failure ((check_main result << 16) | 0x3333), then waits forever.
# Exceptions:
#   - No runtime environment checks are implemented.
# Registers:
#   Establishes the initial stack; no caller register state is preserved.
# =======================================================
    .section .text.start
    .globl _start
_start:
    la sp, __stack_top
    la t0, __bss_start
    la t1, __bss_end
1:
    bgeu t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b
2:
    call check_main
    li t0, 0x100000
    beqz a0, 3f
    slli a0, a0, 16
    li t1, 0x3333
    or a0, a0, t1
    sw a0, 0(t0)
    j 4f
3:
    li t1, 0x5555
    sw t1, 0(t0)
4:
    wfi
    j 4b
