#!/usr/bin/env bash
# Finish the Verilator link. dpi_memutil.o references two SV-exported DPI functions that the Ara
# testbench never uses (it uses VerilatorMemUtil); the newer toolchain no longer drops them, so link
# in no-op stubs (scripts/dpi_stubs.cc).
set -u
cd /mnt/d_ara/hardware/build/verilator
clang++ -c "/mnt/d/AI Thesis/scripts/dpi_stubs.cc" -o dpi_stubs.o || exit 1
rm -f Vara_tb_verilator
make -f Vara_tb_verilator.mk VM_USER_LDLIBS="dpi_stubs.o -lelf" > /mnt/d_ara/verilate_link.log 2>&1
echo "link exit: $?"
tail -n 3 /mnt/d_ara/verilate_link.log
ls -la Vara_tb_verilator
