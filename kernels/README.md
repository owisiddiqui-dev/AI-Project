# RVV 1.0 Kernels

This directory will contain the optimized C/assembly RISC-V Vector (RVV 1.0) kernels to be benchmarked on CVA6 + Ara:
- `conv2d_int8/`: 2D convolution kernel with configurable loop tiling, layout (NHWC/NCHW), and LMUL.
- `depthwise_conv2d_int8/`: Depthwise separable convolution kernel for DS-CNN.
- `fc_int8/`: Fully connected / GEMV int8 kernel.
