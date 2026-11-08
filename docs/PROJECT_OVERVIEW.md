# Project Overview & Technical Specifications

## Executive Summary

Edge AI applications such as always-on keyword spotting, wearable biometric monitoring, and edge computer vision require low-latency, energy-efficient inference under strict compute and power budgets. While open-source RISC-V vector processors provide a customizable, license-free foundation, the impact of model quantization and low-level kernel configurations on vector hardware remains under-explored.

This project investigates the deployment and performance characterization of quantized deep learning models on the **CVA6 processor with the Ara RISC-V Vector (RVV 1.0) coprocessor**. 

Representative models from the **MLPerf Tiny** benchmark suite (Keyword Spotting DS-CNN and Image Classification ResNet-8) are implemented in PyTorch and quantized to Int8 using Quantization-Aware Training (QAT). The core layers (standard convolution, depthwise separable convolution, pointwise 1x1 projection, and fully connected layers) are implemented as optimized RVV 1.0 kernels.

Through cycle-accurate RTL simulation with Verilator, we systematically evaluate how:
1. Vector Register Grouping (LMUL = 1, 2, 4)
2. Loop Tiling Dimensions (spatial vs channel tiling)
3. Activation Layouts (NCHW vs NHWC)

affect execution cycles, memory traffic, and vector lane utilization. A predictive regression cost model is trained on the empirical measurements to accurately predict high-performing kernel configurations across varying input shapes without requiring full RTL simulation runs.

---

## Technical Milestones

| Milestone | Key Deliverables |
| :--- | :--- |
| **M1: Baseline & Toolchain Setup** | Pinned Verilator 5.047, Spike simulator, and LLVM/Clang RVV 1.0 compiler setup; verified baseline `imatmul` on Ara RTL. |
| **M2: Model Quantization (QAT)** | PyTorch MLPerf Tiny implementations (DS-CNN, ResNet-8) with symmetric Int8 tensor quantization. |
| **M3: RVV 1.0 Kernel Library** | Handcrafted vector kernels (`edge_ai_bench.c`) with widening arithmetic, vector accumulators, and golden C reference checking. |
| **M4: Architectural Sweeps** | Systematic sweep across LMUL ($m1, m2, m4$), spatial tiling, and tensor layouts on CVA6 + Ara Verilator simulation. |
| **M5: Predictive Cost Modeling** | Regression model (scikit-learn) mapping layer geometries and vector configurations to simulated cycle counts. |
