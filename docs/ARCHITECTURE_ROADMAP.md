# Architecture & Co-Design Roadmap

## 1. System Architecture

The evaluation platform couples a general-purpose host CPU with a wide vector coprocessor via dedicated high-bandwidth AXI interfaces:

- **Host Processor:** OpenHW Group **CVA6** (64-bit application-class in-order core, RV64GCV).
- **Vector Coprocessor:** PULP Platform **Ara** (RVV 1.0 compliant vector execution engine).
  - Configured with 4 vector lanes and VLEN = 4096 bits.
  - Direct AXI memory port bypassing host L1 cache to access L2/LLC directly.

```
+------------------------------------------------------------------------+
|                               Host CPU                                 |
|                         OpenHW Group CVA6                              |
|                          (64-bit RV64GC)                               |
+-----------------------------------+------------------------------------+
                                    | Control & Instruction Dispatch
                                    v
+------------------------------------------------------------------------+
|                      Vector Accelerator (PULP Ara)                     |
|                                                                        |
|  +--------------------+   +--------------------+   +----------------+  |
|  | Vector Lane 0      |   | Vector Lane 1      |   | Vector Lane 2  |  |
|  +--------------------+   +--------------------+   +----------------+  |
|  +--------------------+   +--------------------+                       |
|  | Vector Lane 3      |   | Vector Load/Store  |   (VLEN = 4096)       |
|  +--------------------+   +--------------------+                       |
+-----------------------------------+------------------------------------+
                                    | 64-bit / 128-bit Wide AXI
                                    v
+------------------------------------------------------------------------+
|                          Memory Subsystem                              |
|                    L2 Cache / Shared LLC DRAM                          |
+------------------------------------------------------------------------+
```

---

## 2. Research & Engineering Phases

### Phase 1: Software & Kernel Optimization (Current)
- Focus: Model- and kernel-level tuning for standard RVV 1.0 hardware.
- Quantization: Int8 Quantization-Aware Training (QAT).
- Kernel Exploration: Loop tiling, register grouping (LMUL sweeps), and activation layout analysis (NCHW vs. NHWC).
- Evaluation: Cycle-accurate Verilator simulations measuring vector utilization, memory stalls, and compute efficiency.

### Phase 2: Hardware/Software Co-Design (Future Extension)
- Focus: Microarchitectural enhancements to Ara's Vector Load-Store Unit (VLSU) and the L2/LLC memory hierarchy.
- Sparsity Acceleration: Hardware support for N:M structured sparsity (e.g. 2:4 sparse loads).
- Prefetching: Stream and tiling-aware prefetcher in the shared LLC (`axi_llc`) to eliminate vector memory starvation on strided kernels.
- Synthesis & Physical Design: Evaluation using Yosys + OpenROAD on the predictive ASAP7 (7nm) cell library for real Area, Power, and Timing (PPA) trade-offs.

---

## 3. Evaluated Workload Topology (MLPerf Tiny)

1. **Keyword Spotting (DS-CNN):**
   - **Initial Conv:** Standard Conv2D, $(10, 4)$ kernel.
   - **Depthwise Separable Blocks:** Depthwise Conv ($3 \times 3$, channel-grouped) + Pointwise Conv ($1 \times 1$).
   - **Classifier:** Global Average Pooling + Fully Connected (GEMV) projection to 12 target classes.
2. **Visual Classification (ResNet-8):**
   - Residual convolutional blocks with skip connections stressing cache locality.
