"""
Model modules
"""
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU

# Residual block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, downsample=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride, kernel_size, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class MLP(nn.Module):
    #~Multilayer Perceptron.
  def __init__(self, in_channels, out_channels=2):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_channels, 12),
        nn.ReLU(),
        nn.Linear(12, 4),
        nn.ReLU(),
        nn.Linear(4, out_channels)
    )

  def forward(self, x):
    #~Forward pass
    return self.layers(x)

class InterNet(nn.Module):
    #~Integrate info from original point 
    #* From Conditional Deformable Image Registration with Convolutional Neural Network
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(
            LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=128)
        )
        self.instance_norm1 = nn.InstanceNorm2d(num_features=256)
        self.instance_norm2 = nn.InstanceNorm2d(num_features=128)


    def forward(self, x, gamma, beta):
        #** x*gamma + beta
        residual = x
        x = self.instance_norm1(x)
        x *= gamma
        x += beta
        x = self.core(x)
        x *= gamma
        x += beta
        x = self.core(x)
        x += residual
        return x



