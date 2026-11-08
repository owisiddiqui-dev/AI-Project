#!/usr/bin/env bash
# Resume the LLVM build with low parallelism (the first attempt was OOM-killed on a ~6 GB WSL VM),
# then build the remaining toolchain pieces (newlib, compiler-rt) without wiping the LLVM build dir.
set -u
git config --global --add safe.directory '*'
cd /mnt/d_ara/toolchain/riscv-llvm/build
ninja -j2 install > /mnt/d_ara/llvm_build2.log 2>&1
rc=$?
echo "llvm main exit: $rc"
if [ $rc -ne 0 ]; then tail -n 20 /mnt/d_ara/llvm_build2.log; exit $rc; fi
cd /mnt/d_ara
make -o toolchain-llvm-main toolchain-llvm > llvm_build3.log 2>&1
echo "toolchain-llvm exit: $?"
tail -n 20 llvm_build3.log
