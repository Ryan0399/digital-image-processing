import torch.nn as nn


class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, 8, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                8, 16, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                16, 32, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 32, Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 64, Output channels: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 128, Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 64, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                16, 8, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 16, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                8, 3, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 8, Output channels: 3
            nn.Tanh(),  # Output activation function for RGB channels
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Decoder forward pass
        y1 = self.deconv1(x5)
        y2 = self.deconv2(y1)
        y3 = self.deconv3(y2)
        y4 = self.deconv4(y3)
        y5 = self.deconv5(y4)

        output = y5

        return output
