#from models.encoder import PositionalEncoding
from .unet_parts import *
class Unet(nn.Module):
    def __init__(self, w_size, in_ch, out_ch, bilinear=True, add_mask=True, *args, **kwargs):
        super().__init__()
        self.n_channels = in_ch # spatial coordinate + mask
        self.n_classes = out_ch
        self.bilinear = bilinear
        self.w_size = w_size

        ch = 32
        self.inc = DoubleConv(self.n_channels, ch*2)
        self.down1 = Down(ch*2, ch*4)
        self.down2 = Down(ch*4, ch*8)
        self.down3 = Down(ch*8, ch*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(ch*8, ch*16 // factor)
        self.up1 = Up(ch*16, ch*16 // factor, bilinear)
        self.up2 = Up(ch*16, ch*8 // factor, bilinear)
        self.up3 = Up(ch*8, ch*4 // factor, bilinear)
        self.up4 = Up(ch*4, ch*2, bilinear)
        self.outc = OutConv(ch*2, self.n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
