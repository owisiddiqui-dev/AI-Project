# Thesis Proposal: Edge AI Acceleration on RISC-V Vector Architectures

## 1. Background & Context
The researcher brings 4 years of industry experience in digital IC design at a fabless semiconductor company, specifically working on:
- **L2 Cache Subsystems:** MSHRs, replacement policies, bandwidth limits, and coherence.
- **Vector Load-Store Units (VLSU):** Memory disambiguation, stride handling, and indexed/gather access.
- **E-Trace:** Hardware trace encoding conforming to RISC-V standards.

To maintain strict IP hygiene and allow open academic publication, this work relies entirely on verified open-source hardware IP:
- **CVA6** (OpenHW Group): 64-bit application core.
- **Ara** (PULP Platform): Scalable vector coprocessor implementing RISC-V Vector Extension (RVV) v1.0.

---

## 2. Two-Stage Research Roadmap

```
+-----------------------------------------------------------------------------------+
| Phase 1: Short Thesis (2 Weeks)                                                   |
| - Focus: Software / Model / Kernel Level                                          |
| - Workloads: MLPerf Tiny (DS-CNN, ResNet-8)                                        |
| - Optimization: Int8 QAT, Loop Tiling, Data Layout, LMUL tuning                   |
| - Simulation: Ara Verilator cycle-accurate simulation                             |
| - Outcome: Identify memory bottlenecks, validate cost model, baseline for RTL    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Phase 2: Master's Thesis Expansion (6–12 Months)                                  |
| - Focus: Hardware / Architecture Co-Design (RTL)                                  |
| - Hardware Changes: Sparse-aware VLSU enhancements (N:M 2:4 sparsity support)     |
| - Cache Subsystem: Tiling-aware prefetcher in LLC (axi_llc)                       |
| - Evaluation: Full PPA with Yosys + OpenROAD on ASAP7 (7nm predictive PDK)        |
| - Publication targets: DATE, ISCAS, CARRV, AICAS                                  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Workload & Model Selection (MLPerf Tiny)

| Task | Target Model | Dataset | Quantization | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Keyword Spotting** | DS-CNN (Depthwise Separable CNN) | Google Speech Commands | int8 (QAT) | Standard edge audio benchmark; depthwise conv stresses memory bandwidth |
| **Image Classification** | ResNet-8 | CIFAR-10 | int8 (QAT) | Standard edge vision benchmark; residual connections test cache reuse |
| *(Optional Stretch)* | FC Autoencoder | ToyADMOS | int8 / int4 | Anomaly detection; evaluates vector linear projection |

---

## 4. Methodology & Tool Stack

- **Model Training:** Python 3.10, PyTorch, [Brevitas](https://github.com/Xilinx/brevitas) for Quantization-Aware Training (QAT).
- **RTL & Simulation:**
  - `pulp-platform/ara` (with CVA6 core submodule)
  - **Verilator** (pinned version built directly from Ara's build system)
  - **Spike** (RISC-V ISA reference simulator)
- **Toolchain:** LLVM/Clang targeting RVV 1.0 (`rv64gcv`).
- **Physical Design (Phase 2):** OpenROAD flow (ORFS) with ASAP7 predictive 7nm PDK.

---

## 5. Key Research Questions

1. **Kernel Sensitivity:** How do vector register allocation (LMUL), loop tiling dimensions, and activation memory layouts (NHWC vs NCHW) impact memory stall cycles on Ara's vector lanes?
2. **Quantization Bottlenecks:** When compute intensity is reduced via int8 quantization, where do memory bottlenecks emerge across different layer types (standard convolution, depthwise convolution, linear projection)?
3. **Analytical vs Learned Cost Models:** Can a lightweight regression cost model predict Pareto-optimal kernel configurations across varying matrix dimensions without requiring exhaustive cycle-accurate Verilator simulations?
