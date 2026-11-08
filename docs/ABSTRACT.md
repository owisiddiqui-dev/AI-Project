# Thesis Abstract

### Title
**Model- and Kernel-Level Optimization of Quantized Edge AI Inference on an Open-Source RISC-V Vector Processor**

### Target Submission
- **Deadline:** 2 Weeks
- **Word Count:** ~185 words (under the 200-word limit)

---

### Abstract Text

Edge AI applications such as always-on keyword spotting, wearable health monitoring, and smart sensors demand low-latency, energy-efficient inference on resource-constrained hardware. Open-source RISC-V vector processors offer a license-free, customizable platform for such workloads, yet the impact of model compression and kernel design choices on their performance remains poorly characterized. This thesis studies the deployment of quantized neural networks on the CVA6 processor with the Ara RISC-V Vector (RVV 1.0) coprocessor. Keyword spotting (DS-CNN) and image classification (ResNet-8) models from the MLPerf Tiny benchmark are trained in PyTorch and quantized to int8 using quantization-aware training. The models' core layers are implemented as RVV kernels, and the effect of loop tiling, data layout, and vector length configuration (LMUL) on cycle count, memory traffic, and vector unit utilization is evaluated through cycle-accurate RTL simulation. A lightweight regression-based cost model is then trained on the collected measurements to predict efficient kernel configurations without exhaustive simulation. The results quantify the accuracy–latency trade-offs of quantization and identify memory-bound layers, providing practical deployment guidelines for low-power, privacy-preserving edge devices and establishing a baseline for future hardware enhancements to the vector memory subsystem.

---

### Two-Week Execution Plan

| Days | Milestones | Key Deliverables |
| :--- | :--- | :--- |
| **Days 1–3** | Environment & Baseline Simulation | Set up CVA6 + Ara with Verilator; verify baseline int8 conv/matmul execution. |
| **Days 2–5** | PyTorch & Quantization (QAT) | Train DS-CNN and ResNet-8 on PyTorch with int8 Quantization-Aware Training. |
| **Days 5–9** | RVV Kernels & Sweeps | Implement core layers (Conv2D, Depthwise-Conv, FC) in RVV; sweep tiling, layout, and LMUL; collect cycle counts & cache misses. |
| **Days 9–10** | Cost Model | Train regression model (scikit-learn) on sweep data to predict optimal kernel configurations. |
| **Days 11–14** | Analysis & Writing | Generate Pareto curves (accuracy vs. latency), discuss bottlenecks, and finalize report. |
