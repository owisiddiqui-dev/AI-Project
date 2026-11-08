import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module):
    """ResNet Basic Block for ResNet-8"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ResNet8(nn.Module):
    """ResNet-8 architecture for Tiny Visual Classification (MLPerf Tiny benchmark)"""
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # Initial 3x3 Conv
        self.conv_in = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 3 Residual Stages (16 -> 32 -> 64)
        self.layer1 = ResNetBasicBlock(16, 16, stride=1)
        self.layer2 = ResNetBasicBlock(16, 32, stride=2)
        self.layer3 = ResNetBasicBlock(32, 64, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
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
    model = ResNet8(num_classes=10, in_channels=3)
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32) # Standard CIFAR-10 shape
    out = model(dummy_input)
    print(f"ResNet-8 model initialized successfully. Output shape: {out.shape}")

    # Verify layer dimensions and int8 weight quantization
    conv1_w = model.layer1.conv1.weight.data
    q_w, scale = quantize_tensor_int8(conv1_w)
    print(f"Layer1 Conv1 weight: shape={conv1_w.shape}, scale={scale:.6f}, range=[{q_w.min()}, {q_w.max()}]")
