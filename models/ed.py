from torch import nn

from .networks import Encoder, Decoder,\
            GaussianLayer2D, GaussianLayer3D,\
            SobelLayer2D


class EDSeg(nn.Module):
    def __init__(self, opt):
        super(EDSeg, self).__init__()
        self.encoder = Encoder(opt)
        expansion = self.encoder.expansion
        self.decoder = Decoder(opt, expansion)
        if opt.dim == 2:
            self.gaussian = GaussianLayer2D(opt.image_chan, opt.image_chan, kernel_size=5,
                                            sigma=opt.sigma, groups=opt.image_chan)
        else:
            self.gaussian = GaussianLayer3D(opt.image_chan, opt.image_chan, kernel_size=5,
                                            sigma=opt.sigma, groups=opt.image_chan)

        self.sobel = SobelLayer2D(opt.image_chan, opt.image_chan, groups=opt.image_chan)

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        x = self.decoder(c1, c2, c3, c4)
        x = self.gaussian(x)
        return x
