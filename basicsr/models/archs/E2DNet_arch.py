# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
import math
from basicsr.models.archs.E2DNet_util import LKNAFBlock, rgb2lab, lab2rgb, ReFine

class E2DNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, refine_nums=2, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1,
                               stride=1, groups=1, bias=True)

        self.l_end = nn.Conv2d(in_channels=width, out_channels=img_channel // 3, kernel_size=3, padding=1,
                               stride=1, groups=1, bias=True)

        self.ab_end = nn.Conv2d(in_channels=width, out_channels=img_channel - 1, kernel_size=3, padding=1,
                                stride=1, groups=1, bias=True)

        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_blk = nn.ModuleList()
        
        self.l_ups = nn.ModuleList()
        self.l_skips = nn.ModuleList()
        self.l_decoders = nn.ModuleList()

        self.ab_ups = nn.ModuleList()
        self.ab_skips = nn.ModuleList()
        self.ab_decoders = nn.ModuleList()

        chan = width
        Block = LKNAFBlock
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(*[Block(chan, scale=i+j+1) for j in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        enc_lay = sum(enc_blk_nums)
        
        self.middle_blk = nn.Sequential(*[Block(chan, scale=enc_lay+j+1) for j in range(middle_blk_num)])
        self.conv_l = ReFine(chan)
        self.conv_ab = ReFine(chan)
        
        enc_lay += middle_blk_num
        for i, num in enumerate(dec_blk_nums):
            self.l_ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            self.ab_ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.l_decoders.append(nn.Sequential(*[Block(chan, scale=enc_lay+i+j+1) for j in range(num)]))
            self.ab_decoders.append(nn.Sequential(*[Block(chan, scale=enc_lay+i+j+1) for j in range(num)]))
            self.l_skips.append(ReFine(chan))
            self.ab_skips.append(ReFine(chan))
            
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        new_inp, l, ab = rgb2lab(inp)
        x = self.intro(new_inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blk(x)
        x_l = self.conv_l(x)
        x_ab = self.conv_ab(x) 

        for decoder, up, skip, enc_skip in zip(self.l_decoders, self.l_ups, self.l_skips, encs[::-1]):
            x_l = up(x_l)
            x_l = skip(enc_skip) + x_l
            x_l = decoder(x_l)
        for decoder, up, skip, enc_skip in zip(self.ab_decoders, self.ab_ups, self.ab_skips, encs[::-1]):
            x_ab = up(x_ab)
            x_ab = skip(enc_skip) + x_ab
            x_ab = decoder(x_ab)

        x_l = self.l_end(x_l) + l
        x_ab = self.ab_end(x_ab) + ab
        out = lab2rgb(x_l, x_ab)

        return x_l[:, :, :H, :W], x_ab[:, :, :H, :W], out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class E2DNetLocal(Local_Base, E2DNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        E2DNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

