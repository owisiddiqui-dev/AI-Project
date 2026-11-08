import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConvBlock(nn.Module):
    """Depthwise Separable Convolution Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Pointwise
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x

class DSCNN(nn.Module):
    """DS-CNN architecture for Edge Keyword Spotting (MLPerf Tiny benchmark)"""
    def __init__(self, num_classes=12):
        super().__init__()
        # First Standard Conv Layer
        self.init_conv = nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 1), padding=(5, 1), bias=False)
        self.bn_init = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Depthwise Separable Blocks
        self.blocks = nn.Sequential(
            DSConvBlock(64, 64),
            DSConvBlock(64, 64),
            DSConvBlock(64, 64),
            DSConvBlock(64, 64)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn_init(self.init_conv(x)))
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def quantize_tensor_int8(tensor):
    """Symmetric Int8 Quantization"""
    max_val = torch.max(torch.abs(tensor))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return q_tensor, float(scale)

if __name__ == "__main__":
    model = DSCNN(num_classes=12)
    model.eval()
    dummy_input = torch.randn(1, 1, 49, 10) # 49 frames, 10 MFCC features
    out = model(dummy_input)
    print(f"DS-CNN Model successfully constructed. Output shape: {out.shape}")
    
    # Verify quantization on first depthwise layer
    dw_weight = model.blocks[0].dw.weight.data
    q_weight, scale = quantize_tensor_int8(dw_weight)
    print(f"Layer dw.weight: shape={dw_weight.shape}, scale={scale:.6f}, int8 range=[{q_weight.min()}, {q_weight.max()}]")
