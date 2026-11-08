# Models Directory

This directory contains the MLPerf Tiny deep learning models implemented in PyTorch with symmetric Int8 tensor quantization for Edge AI deployment on RISC-V Vector hardware:

- `ds_cnn_quant.py`: Depthwise Separable CNN for Keyword Spotting (Speech Commands dataset), layer definitions, and Int8 tensor quantization.
- `resnet8_quant.py`: Compact residual neural network (ResNet-8) for tiny visual classification (CIFAR-10 / Visual Wake Words), layer definitions, and Int8 tensor quantization.
- `cost_model.py`: Analytical and regression cost models predicting execution cycles across vector configurations (LMUL, tiling dimensions, tensor layout).
