import torch
from torch import nn
import torch.nn.functional as F

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
        self.up1 = nn.ConvTranspose2d(512 * expansion, 256, 4, 3)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.up4 = nn.ConvTranspose2d(64, 1, 3, 2)

    def forward(self, c1, c2, c3, c4):
         x = self.up1(c4)
         x = self.up2(x)
         x = self.up3(x)
         x = self.up4(x)
         x = F.interpolate(x, size=self.output_size, mode='bilinear',
                            align_corners=False)
         return x
