import torch.nn as nn
import torch


class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 3, 输出通道: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 64, 输出通道: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                128, 256, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 128, 输出通道: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                256, 512, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 256, 输出通道: 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 512, 输出通道: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 128, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 512, 输出通道: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 256, 输出通道: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 3, kernel_size=4, stride=2, padding=1
            ),  # 输入通道: 128, 输出通道: 3
            nn.Tanh(),  # 输出激活函数，适用于 RGB 通道
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Decoder forward pass with skip connections
        y1 = self.deconv1(x4)
        y2 = self.deconv2(torch.cat([y1, x3], dim=1))
        y3 = self.deconv3(torch.cat([y2, x2], dim=1))
        y4 = self.deconv4(torch.cat([y3, x1], dim=1))

        output = y4

        return output
