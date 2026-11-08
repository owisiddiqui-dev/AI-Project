#!/usr/bin/env bash
set -e

REPO_DIR="/mnt/d/AI Thesis"
ARA_APPS="/mnt/d_ara/apps"

echo "=== Syncing Edge AI kernels to Ara application tree ==="
mkdir -p "${ARA_APPS}/edge_ai_kernels"
cp "${REPO_DIR}/kernels/edge_ai_bench.c" "${ARA_APPS}/edge_ai_kernels/"
cp "${REPO_DIR}/kernels/edge_ai_bench.h" "${ARA_APPS}/edge_ai_kernels/"
cp "${REPO_DIR}/apps/edge_ai_kernels/main.c" "${ARA_APPS}/edge_ai_kernels/"

echo "=== Compiling edge_ai_kernels binary ==="
cd "${ARA_APPS}"
rm -f bin/edge_ai_kernels
make bin/edge_ai_kernels

echo "=== Executing on CVA6 + Ara Verilator Model ==="
cd /mnt/d_ara/hardware
./build/verilator/Vara_tb_verilator -l ram,../apps/bin/edge_ai_kernels,elf
