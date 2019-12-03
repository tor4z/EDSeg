from torch import nn

from .networks import Encoder, Decoder


class EDSeg(nn.Module):
    def __init__(self, opt):
        super(EDSeg, self).__init__()
        self.encoder = Encoder(opt)
        expansion = self.encoder.expansion
        self.decoder = Decoder(opt, expansion)

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        x = self.decoder(c1, c2, c3, c4)
        return x
