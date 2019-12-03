import torch
from torch import nn


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
