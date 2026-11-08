# Edge AI on RISC-V Vector Processor (CVA6 + Ara)

This repository contains the codebase, research documentation, and experiments for the thesis:
> **"Model- and Kernel-Level Optimization of Quantized Edge AI Inference on an Open-Source RISC-V Vector Processor"**

Imported from conversation: [Thesis Ideation & Environment Setup](conversation://6ea18a81-cca2-426e-837c-a8bde090512f).

---

## 📌 Project Overview

- **Context & Domain:** Hardware/Software co-design for Edge AI using open-source RISC-V microarchitecture.
- **Background:** IC design (L2 Cache, Vector Load/Store Unit - VLSU, and E-Trace in RISC-V cores).
- **Core Architecture:** [CVA6](https://github.com/openhwgroup/cva6) (64-bit application-class RISC-V core) coupled with [PULP Ara](https://github.com/pulp-platform/ara) (RVV 1.0 vector coprocessor).
- **Simulation Platform:** Cycle-accurate RTL simulation with **Verilator**, supplemented by Spike ISA simulator.

---

## 🗂️ Documentation & Milestones

1. **[Thesis Abstract](docs/ABSTRACT.md)**: Ready-to-submit abstract (< 200 words) for the professor (2-week deadline).
2. **[Thesis Proposal & Roadmap](docs/THESIS_PROPOSAL.md)**:
   - **Phase 1 (2-Week Short Thesis):** Model- and kernel-level optimization of int8 quantized edge AI models (DS-CNN, ResNet-8 from MLPerf Tiny), sweeping loop tiling, data layout, and LMUL configurations with a regression-based cost model.
   - **Phase 2 (Master's Thesis RTL Co-Design):** Sparsity- and quantization-aware hardware co-design (custom sparse/compressed loads in Ara VLSU, LLC prefetching, and PPA analysis on ASAP7).
3. **[Environment Setup & Status](docs/ENVIRONMENT_SETUP.md)**: Complete record of WSL2 (Ubuntu 22.04 LTS), toolchain dependencies, swap configuration, and build steps.

---

## 🚀 Environment Quick Start

The host environment runs on Windows 11 with **WSL2 (Ubuntu 22.04 LTS)**.

```bash
# Inside WSL2 (Ubuntu 22.04):
# 1. Enter the project directory on Drive D:
cd "/mnt/d/AI Thesis/ara"

# 2. Build Verilator (pinned version in Ara repo)
make verilator

# 3. Build Spike and the RISC-V Vector LLVM Toolchain
make riscv-isa-sim
make toolchain-llvm
```
*(Helper scripts are available in the `scripts/` directory).*

---

## 📂 Repository Structure

```text
├── docs/
│   ├── ABSTRACT.md            # Submitted 2-week thesis abstract
│   ├── THESIS_PROPOSAL.md     # Full proposal, research questions & methodology
│   └── ENVIRONMENT_SETUP.md   # Setup status, tool versions, and instructions
├── scripts/
│   └── setup_ara.sh           # Automated script for WSL setup and builds
├── models/                    # PyTorch / Brevitas training scripts (MLPerf Tiny)
├── kernels/                   # Custom RVV 1.0 C/assembly kernels
└── README.md
```
