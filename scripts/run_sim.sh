#!/usr/bin/env bash
# Run an app on the CVA6+Ara Verilator model. Usage: run_sim.sh <app>   (default hello_world)
app=${1:-hello_world}
cd /mnt/d_ara/hardware
timeout 900 ./build/verilator/Vara_tb_verilator -l ram,../apps/bin/$app,elf 2>&1 | tail -n 30
echo "sim exit: ${PIPESTATUS[0]}"
