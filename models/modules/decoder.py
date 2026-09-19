from torch import nn


def convolution(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1))
