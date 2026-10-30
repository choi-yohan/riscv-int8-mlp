"""Bound execution time so a bare-metal failure cannot hang a test run."""
import argparse
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--qemu', default='qemu-system-riscv32')
p.add_argument('--elf', default='build/check.elf')
a = p.parse_args()
result = subprocess.run([a.qemu, '-M', 'virt', '-m', '16M', '-smp', '1',
                         '-bios', 'none', '-kernel', a.elf, '-nographic',
                         '-monitor', 'none', '-serial', 'none'], timeout=30)
if result.returncode:
    raise SystemExit(f'RV32 execution failed: {result.returncode}')
print('RV32IM: dot-product edge cases and full MLP golden fixtures passed')
