# RISC-V Vector Edge AI Acceleration (CVA6 + Ara)

An open-source performance benchmarking and co-design project evaluating quantized (`int8`) deep neural network inference on an open-source RISC-V Vector coprocessor.

---

## 📌 Project Overview

Edge AI applications such as always-on keyword spotting, wearable biometric monitoring, and edge computer vision require low-latency, energy-efficient inference under strict compute and power budgets. 

This project explores model- and kernel-level optimizations of quantized neural networks targeting the open-source **[CVA6](https://github.com/openhwgroup/cva6)** 64-bit application-class processor coupled with the **[PULP Ara](https://github.com/pulp-platform/ara)** vector coprocessor (RISC-V Vector Extension v1.0).

### Key Objectives
- **Quantization Optimization:** Train edge models (MLPerf Tiny benchmark suite) using Int8 Quantization-Aware Training (QAT).
- **Handcrafted RVV 1.0 Kernels:** Implement vector-optimized kernels (Depthwise Conv2D, Pointwise Conv 1x1, GEMV) leveraging widening multiply-accumulates and vector register groups (LMUL sweeps).
- **Architectural Bottleneck Profiling:** Characterize memory bandwidth, vector lane utilization, and execution latency using cycle-accurate RTL simulation with Verilator.
- **Analytical & Predictive Modeling:** Formulate regression cost models to predict optimal loop tiling and LMUL configurations without exhaustive cycle-accurate simulation.

---

## 🏛️ Architecture & Platform

- **Core:** CVA6 (64-bit RISC-V `rv64gcv` application core, SV39 MMU).
- **Vector Coprocessor:** PULP Ara (RVV 1.0 compliant coprocessor with 4 vector lanes, VLEN = 4096 bits).
- **Memory Subsystem:** AXI bus interface bypassing L1 to connect directly to L2/LLC memory.
- **Toolchain:** LLVM/Clang (RVV 1.0 intrinsics), Spike ISA simulator, and Verilator 5.047.

---

## 📂 Repository Structure

```text
├── docs/
│   ├── PROJECT_OVERVIEW.md       # Technical summary, benchmarks & evaluation plan
│   ├── ARCHITECTURE_ROADMAP.md   # Architectural design, memory hierarchy & roadmap
│   └── ENVIRONMENT_SETUP.md      # Toolchain build guide, WSL2 setup & reproducibility
├── scripts/
│   ├── run_as_owi.sh             # WSL2 environment wrapper
│   ├── run_sim.sh                # Verilator simulation runner
│   ├── setup_ara.sh              # Ara & toolchain setup automation
│   └── build_*.sh                # Individual toolchain build scripts
├── models/
│   ├── README.md                 # MLPerf Tiny model definitions
│   └── ds_cnn_quant.py           # PyTorch DS-CNN with int8 symmetric quantization
├── kernels/
│   ├── README.md                 # RVV 1.0 kernel specifications
│   ├── depthwise_conv2d_int8.c   # Int8 3x3 depthwise convolution (RVV 1.0)
│   ├── edge_ai_bench.h           # Kernel headers and golden reference signatures
│   └── edge_ai_bench.c           # Vectorized Depthwise, Pointwise, and GEMV kernels
└── README.md
```

---

## 🚀 Quick Start & Reproduction

### Prerequisites
- OS: Linux or Windows 11 with WSL2 (Ubuntu 22.04 LTS).
- System Dependencies: GCC, Clang, CMake, Ninja, Device-Tree-Compiler, Python 3.10+, PyTorch.

### 1. Build Hardware & Toolchains
Follow the step-by-step instructions in [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md):
```bash
# Clone and build toolchain (Verilator, Spike, RVV LLVM)
bash scripts/setup_ara.sh
```

### 2. Run Edge AI Simulation
Execute the edge AI kernel suite on the cycle-accurate CVA6 + Ara Verilator model:
```bash
# Inside WSL2
bash scripts/run_as_owi.sh scripts/run_sim.sh edge_ai_kernels
```

---

## 📊 Workloads (MLPerf Tiny)

| Task | Target Model | Dataset | Quantization | Primary Hardware Stress |
| :--- | :--- | :--- | :--- | :--- |
| **Keyword Spotting** | DS-CNN (Depthwise Separable CNN) | Speech Commands | Int8 (QAT) | Vector memory bandwidth, small spatial tiling |
| **Image Classification** | ResNet-8 | CIFAR-10 | Int8 (QAT) | Cache reuse, 3x3 channel-tiled convolution |
| **Edge Anomaly Detection** | FC Autoencoder | ToyADMOS | Int8 | Matrix-vector linear projection throughput |
