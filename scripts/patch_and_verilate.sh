#!/usr/bin/env bash
# Apply the repo's tech_cells_generic SRAM patch (adds the DPI exports Verilator's memutil needs),
# then rebuild the Verilator model with the stub-free link and run hello_world.
set -u
git config --global --add safe.directory '*'
cd /mnt/d_ara/hardware
git -C deps/tech_cells_generic apply --check ../../patches/0001-tech-cells-generic-sram.patch 2>/dev/null \
  && make apply-patches
git -C deps/tech_cells_generic status --short
make verilate > /mnt/d_ara/verilate_build.log 2>&1; echo "verilate exit: $?"
tail -n 5 /mnt/d_ara/verilate_build.log
ls -la build/verilator/Vara_tb_verilator
