import torch
import torch.nn as nn
import torch.nn.functional as F

# 殘差塊（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保留輸入
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity  # 殘差連接

# SRResNet 模型
class SRResNet(nn.Module):
    def __init__(self, num_residuals=32, upscale_factor=3):
        super(SRResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=9, padding=4)
        self.res_blocks = nn.Sequential(*[ResidualBlock(16) for _ in range(num_residuals)])
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # 上採樣（PixelShuffle）
        self.upsample = nn.Sequential(
            nn.Conv2d(16, 16 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Conv2d(16, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 初始卷積層
        residual = self.res_blocks(x)  # 殘差塊
        x = self.conv2(residual) + x  # 殘差連接
        x = self.upsample(x)  # 上採樣
        x = self.conv3(x)  # 輸出
        return torch.clamp(x, 0.0, 1.0)  # 確保輸出範圍在 [0, 1] 之間
