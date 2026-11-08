#!/usr/bin/env bash
# Run as user owi (not root) so git ownership checks pass.
set -u
git config --global --add safe.directory '*'
cd /mnt/d_ara
rm -rf install/riscv-isa-sim install/riscv-isa-sim-mod
make riscv-isa-sim > spike_build.log 2>&1
echo "exit: $?"
tail -n 15 spike_build.log
