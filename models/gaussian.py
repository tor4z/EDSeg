import torch
from torch import nn
from scipy import ndimage
import numpy as np



class GaussianLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma,
                stride=1, padding=0, bias=None, groups=3):
        super(GaussianLayer2D, self).__init__()
        if isinstance(kernel_size, tuple):
            origin_kernel = np.zeros(kernel_size)
            center = kernel_size[0] // 2
            origin_kernel[center, center] = 1
        else:
            origin_kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            origin_kernel[center, center] = 1
        kernel = ndimage.gaussian_filter(origin_kernel, sigma=sigma)
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(out_channels, in_channels, 1, 1)

        self.register_buffer('gaussian_kernel', kernel)

        self.pad = nn.ReflectionPad2d(center)
        self.gaussian = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=bias, groups=groups)

        self.weights_init()

    def forward(self, x):
        x = self.pad(x)
        x = self.gaussian(x)
        return x

    def weights_init(self):
        self.gaussian.weight.data = self.gaussian_kernel
        self.gaussian.weight.requires_grad = False


class GaussianLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma,
                stride=1, padding=0, bias=None, groups=3):
        super(GaussianLayer3D, self).__init__()
        if isinstance(kernel_size, tuple):
            origin_kernel = np.zeros(kernel_size)
            center = kernel_size[0] // 2
            origin_kernel[center, center, center] = 1
        else:
            origin_kernel = np.zeros((kernel_size, kernel_size, kernel_size))
            center = kernel_size // 2
            origin_kernel[center, center, center] = 1
        kernel = ndimage.gaussian_filter(origin_kernel, sigma=sigma)
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(out_channels, in_channels, 1, 1, 1)

        self.register_buffer('gaussian_kernel', kernel)

        self.pad = nn.ReflectionPad3d(center)
        self.gaussian = nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=bias, groups=groups)

        self.weights_init()

    def forward(self, x):
        x = self.pad(x)
        x = self.gaussian(x)
        return x

    def weights_init(self):
        self.gaussian.weight.data = self.gaussian_kernel
        self.gaussian.weight.requires_grad = False