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
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256 * expansion, 256, 4, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, 1),
            )

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


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()

        nets = [
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=3),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=5, stride=2)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, kernel_size=5, stride=2)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 128, kernel_size=3, stride=2)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.AdaptiveAvgPool2d(1, 1)
            ]

        self.net = nn.Sequential(*nets)

        self.classifier = [
            nn.Linear(128, 1),
            nn.Sigmoid()]

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, -1)
        return self.classifier(x)

