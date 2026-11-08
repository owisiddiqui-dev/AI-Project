#!/usr/bin/env bash
# Build the RVV-capable LLVM toolchain. Run as user owi, with /mnt/d_ara bind-mounted.
set -u
git config --global --add safe.directory '*'
cd /mnt/d_ara
make toolchain-llvm > llvm_build.log 2>&1
echo "exit: $?"
tail -n 20 llvm_build.log
