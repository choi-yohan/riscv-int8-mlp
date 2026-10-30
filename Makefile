CC = cc
PYTHON ?= python3
RVCC ?= clang
RVLD ?= ld.lld
QEMU ?= qemu-system-riscv32
CFLAGS = -O2 -std=c11 -Wall -Wextra -Werror -Iinclude
RVFLAGS = --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -fno-stack-protector -msmall-data-limit=0 -Iinclude

.PHONY: all test rv32-check clean
all: build/mlp build/libmlp.so
build:
	mkdir -p build
build/mlp: src/cli.c src/mlp.c include/mlp.h include/mlp_layout.h | build
	$(CC) $(CFLAGS) src/cli.c src/mlp.c -o $@
build/libmlp.so: src/mlp.c include/mlp.h include/mlp_layout.h | build
	$(CC) $(CFLAGS) -fPIC -shared src/mlp.c -o $@
test: all
	$(PYTHON) -m unittest discover -s tests -v
build/mlp_rv.o: src/mlp.c include/mlp.h include/mlp_layout.h | build
	$(RVCC) $(RVFLAGS) -DMLP_USE_RV32 -c $< -o $@
build/dot_rv.o: src/dot_rv32.s | build
	$(RVCC) $(RVFLAGS) -x assembler-with-cpp -c $< -o $@
build/predict_rv.o: src/predict_rv32_legacy.s include/mlp_layout.h | build
	$(RVCC) $(RVFLAGS) -x assembler-with-cpp -c $< -o $@
build/start.o: rv32/start.s | build
	$(RVCC) $(RVFLAGS) -x assembler-with-cpp -c $< -o $@
build/check.o: rv32/check.c build/fixture.h include/mlp.h | build
	$(RVCC) $(RVFLAGS) -Ibuild -c $< -o $@
build/runtime.o: rv32/runtime.c | build
	$(RVCC) $(RVFLAGS) -c $< -o $@
build/check.elf: build/start.o build/mlp_rv.o build/dot_rv.o build/predict_rv.o build/check.o build/runtime.o rv32/link.ld
	$(RVLD) -T rv32/link.ld build/start.o build/mlp_rv.o build/dot_rv.o build/predict_rv.o build/check.o build/runtime.o -o $@
rv32-check: build/check.elf
	$(PYTHON) tools/run_qemu.py --qemu $(QEMU) --elf $<
clean:
	rm -rf build
