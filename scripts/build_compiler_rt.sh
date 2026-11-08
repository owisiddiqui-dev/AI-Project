#!/usr/bin/env bash
# Build compiler-rt builtins only (bare-metal). Same flags as `make toolchain-llvm-rt`, plus extra
# components disabled because newer LLVM pulls in sanitizer_common (unsupported on bare-metal).
set -u
git config --global --add safe.directory '*'
L=/mnt/d_ara/install/riscv-llvm
cd /mnt/d_ara/toolchain/riscv-llvm/compiler-rt
rm -rf build && mkdir -p build && cd build
cmake /mnt/d_ara/toolchain/riscv-llvm/compiler-rt -G Ninja \
  -DCMAKE_INSTALL_PREFIX=$L \
  -DCMAKE_C_COMPILER_TARGET=riscv64-unknown-elf \
  -DCMAKE_ASM_COMPILER_TARGET=riscv64-unknown-elf \
  -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
  -DCOMPILER_RT_BAREMETAL_BUILD=ON \
  -DCOMPILER_RT_BUILD_BUILTINS=ON \
  -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
  -DCOMPILER_RT_BUILD_MEMPROF=OFF \
  -DCOMPILER_RT_BUILD_PROFILE=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
  -DCOMPILER_RT_BUILD_XRAY=OFF \
  -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
  -DCOMPILER_RT_BUILD_ORC=OFF \
  -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
  -DCMAKE_C_COMPILER_WORKS=1 -DCMAKE_CXX_COMPILER_WORKS=1 -DCMAKE_SIZEOF_VOID_P=4 \
  -DCMAKE_C_COMPILER=$L/bin/clang \
  -DCMAKE_C_FLAGS="-march=rv64gc -mabi=lp64d -mno-relax -mcmodel=medany" \
  -DCMAKE_ASM_FLAGS="-march=rv64gc -mabi=lp64d -mno-relax -mcmodel=medany" \
  -DCMAKE_AR=$L/bin/llvm-ar -DCMAKE_NM=$L/bin/llvm-nm -DCMAKE_RANLIB=$L/bin/llvm-ranlib \
  -DLLVM_CONFIG_PATH=$L/bin/llvm-config || exit 1
ninja -j2 install || exit 1
[ -e $L/lib/clang/20/lib ] || ln -s $L/lib/linux $L/lib/clang/20/lib
echo "compiler-rt done"
ls $L/lib/linux | head; $L/bin/clang --version | head -n 1
