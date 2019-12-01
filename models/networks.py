import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from .resnet import resnet


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.base = resnet(opt)
        self.expansion = self.base.expansion

    def forward(self, x):
        c1, c2, c3, c4 = self.base(x)
        return c1, c2, c3, c4


class Decoder(nn.Module):
    def __init__(self, opt, expansion):
        super(Decoder, self).__init__()
        self.output_size=(opt.image_x, opt.image_y)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256 * expansion, 256, 4, 3),
            # nn.BatchNorm2d(256),
            # nn.ReLU()
            )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 3),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
            )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, 1),
            # nn.Sigmoid()
            )
        # self.up4 = nn.ConvTranspose2d(64, 1, 3, 2)

    def forward(self, c1, c2, c3, c4):
        # c1: torch.Size([8, 64, 128, 128])
        # c2: torch.Size([8, 128, 64, 64])
        # c3: torch.Size([8, 256, 32, 32])
        # c4: torch.Size([8, 512, 16, 16])

        x = self.up1(c3)
        # print(x.shape)
        x = self.up2(x)
        # print(x.shape)
        x = self.up3(x)
        # print(x.shape)
        # x = self.up4(x)
        x = self.out(x)         # torch.Size([8, 1, 590, 590])
        # print(x.shape)
        x = F.interpolate(x, size=self.output_size, mode='bilinear',
                           align_corners=False)
        return x


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


class SobelLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                padding=1, bias=None, groups=3):
        super(SobelLayer2D, self).__init__()
        kernel_h = [[-1, -2, -1],
                    [0,   0,  0],
                    [1,   2,  1]]
        kernel_v = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        
        kernel_h = torch.Tensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_h = kernel_h.repeat(out_channels, in_channels, 1, 1)

        kernel_v = torch.Tensor(kernel_v).unsqueeze(0).unsqueeze(0)
        kernel_v = kernel_v.repeat(out_channels, in_channels, 1, 1)
        
        self.register_buffer('kernel_h', kernel_h)
        self.register_buffer('kernel_v', kernel_v)

        self.sobel_h = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=padding, bias=bias, groups=groups)
        self.sobel_v = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=padding, bias=bias, groups=groups)
        self.init_weight()

    def forward(self, x):
        x = self.sobel_h(x)
        x = self.sobel_v(x)
        return x

    def init_weight(self):
        self.sobel_h.weight.data = self.kernel_h
        self.sobel_h.weight.requires_grad = False

        self.sobel_v.weight.data = self.kernel_v
        self.sobel_v.weight.requires_grad = False


class PrewittLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                padding=1, bias=None, groups=3):
        super(PrewittLayer2D, self).__init__()
        kernel_h = [[-1, -1, -1],
                    [0,   0,  0],
                    [1,   1,  1]]
        kernel_v = [[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]]
        
        kernel_h = torch.Tensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_h = kernel_h.repeat(out_channels, in_channels, 1, 1)

        kernel_v = torch.Tensor(kernel_v).unsqueeze(0).unsqueeze(0)
        kernel_v = kernel_v.repeat(out_channels, in_channels, 1, 1)
        
        self.register_buffer('kernel_h', kernel_h)
        self.register_buffer('kernel_v', kernel_v)

        self.prewitt_h = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=padding, bias=bias, groups=groups)
        self.prewitt_v = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=padding, bias=bias, groups=groups)
        self.init_weight()

    def forward(self, x):
        x = self.prewitt_h(x)
        x = self.prewitt_v(x)
        return x

    def init_weight(self):
        self.prewitt_h.weight.data = self.kernel_h
        self.prewitt_h.weight.requires_grad = False

        self.prewitt_v.weight.data = self.kernel_v
        self.prewitt_v.weight.requires_grad = False

 
class RobertsLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                padding=1, bias=None, groups=3):
        super(RobertsLayer2D, self).__init__()
        kernel_a = [[-1, 0],
                    [0,  1]]
        kernel_b = [[0, -1],
                    [1, 0]]
        
        kernel_a = torch.Tensor(kernel_a).unsqueeze(0).unsqueeze(0)
        kernel_a = kernel_a.repeat(out_channels, in_channels, 1, 1)

        kernel_b = torch.Tensor(kernel_b).unsqueeze(0).unsqueeze(0)
        kernel_b = kernel_b.repeat(out_channels, in_channels, 1, 1)
        
        self.register_buffer('kernel_a', kernel_a)
        self.register_buffer('kernel_b', kernel_b)

        self.roberts_a = nn.Conv2d(in_channels, out_channels, 2, stride=stride,
                                padding=padding, bias=bias, groups=groups)
        self.roberts_b = nn.Conv2d(in_channels, out_channels, 2, stride=stride,
                                padding=padding, bias=bias, groups=groups)
        self.init_weight()

    def forward(self, x):
        x = self.roberts_a(x)
        x = self.roberts_b(x)
        return x

    def init_weight(self):
        self.roberts_a.weight.data = self.kernel_a
        self.roberts_a.weight.requires_grad = False

        self.roberts_b.weight.data = self.kernel_b
        self.roberts_b.weight.requires_grad = False