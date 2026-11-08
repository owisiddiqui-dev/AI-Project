#!/usr/bin/env bash
# ==============================================================================
# Setup script for PULP Ara & CVA6 with Verilator and LLVM inside WSL2
# Run inside Ubuntu 22.04: bash scripts/setup_ara.sh
# ==============================================================================

set -euo pipefail

# Use the bind mount path to avoid whitespace issues in GNU Make
ARA_DIR="/mnt/d_ara"

echo "=== [1/4] Checking/Cloning pulp-platform/ara ==="
if [ ! -d "$ARA_DIR/.git" ]; then
    echo "Cloning Ara with submodules into $ARA_DIR..."
    git clone --recursive https://github.com/pulp-platform/ara.git "$ARA_DIR"
else
    echo "Ara repository already exists at $ARA_DIR. Updating submodules..."
    cd "$ARA_DIR"
    git submodule update --init --recursive
fi

cd "$ARA_DIR"

echo "=== [2/4] Building Verilator ==="
make verilator

echo "=== [3/4] Building Spike ISA Simulator ==="
make riscv-isa-sim

echo "=== [4/4] Building LLVM Toolchain for RVV 1.0 ==="
echo "Note: This step may take 1-2 hours depending on CPU performance."
make toolchain-llvm

echo "=== Setup Complete! ==="
echo "You can now run test benchmarks in ~/ara/apps"
