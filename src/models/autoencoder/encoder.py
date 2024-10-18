import math
import torch
import torch.nn as nn
from src.models.autoencoder import ConvLayer2d, ConvResBlock2d


class Encoder(nn.Module):
    def __init__(self, in_channel, in_res, out_res, ch_mul=64, ch_max=32, lasts_ch=32, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.conv_in = ConvLayer2d(in_channel=in_channel, out_channel=ch_mul, kernel_size=3)

        # each resblock will half the resolution and double the number of features (until a maximum of ch_max)
        self.layers = []
        in_channels = ch_mul
        for i in range(log_size_in, log_size_out, -1):
            out_channels = int(min(in_channels * 2, ch_max))
            self.layers.append(ConvResBlock2d(in_channel=in_channels, out_channel=out_channels, downsample=True))
            in_channels = out_channels

        self.layers = nn.Sequential(*self.layers)
        self.conv_out = ConvLayer2d(in_channel=in_channels, out_channel=lasts_ch, kernel_size=1, activate=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        x = self.conv_out(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class Decoder(nn.Module):
    def __init__(self, in_channel, in_res, out_res, ch_mul=64, ch_max=32, lasts_ch=32, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.conv_in = ConvLayer2d(in_channel=in_channel, out_channel=ch_mul, kernel_size=3)

        # each resblock will half the resolution and double the number of features (until a maximum of ch_max)
        self.layers = []
        in_channels = ch_mul
        for i in range(log_size_in, log_size_out):
            out_channels = int(min(in_channels * 2, ch_max))
            self.layers.append(ConvResBlock2d(in_channel=in_channels, out_channel=out_channels, upsample=True))
            in_channels = out_channels

        self.layers = nn.Sequential(*self.layers)
        self.conv_out = ConvLayer2d(in_channel=in_channels, out_channel=lasts_ch, kernel_size=1, activate=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        x = self.conv_out(x)
        return x
