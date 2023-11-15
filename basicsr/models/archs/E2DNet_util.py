import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.models.archs.arch_util import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):
        x = inp

        x = self.conv1(self.norm1(x))
        x = self.conv2(x)
        x = self.sg(x)
        
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class LKNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, scale=1):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.convdw1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)
        self.convdw2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=7, padding=3, stride=1, groups=c, bias=True)
    
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )


        self.norm2 = LayerNorm2d(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.scale = scale

        
    def forward(self, inp):

        scale_t = 1. / math.sqrt(self.scale)
        # scale_l = 1.0
        # scale_c = 0.3
        x = inp
        x = self.conv1(self.norm1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = self.convdw1(x1) * self.convdw2(x2)
        x = self.conv3(x)
        y = inp + x * scale_t

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv5(x)
        return y + x * scale_t

class ReFine(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x 


# Color conversion code
def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if (rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2xyz')
    # embed()
    return out


def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if (rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055 * (rgb ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)

    # if(torch.sum(torch.isnan(rgb))>0):
    #     print('xyz2rgb')
    #     embed()
    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if (xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if (xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    if (z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if (out.is_cuda):
        mask = mask.cuda()

    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc

    return out


def rgb2lab(rgb):
    l_cent = 50.
    l_norm = 100.
    ab_norm = 110.
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - l_cent) / l_norm
    ab_rs = lab[:, 1:, :, :] / ab_norm
    out = torch.cat((l_rs, ab_rs), dim=1)

    return out, l_rs, ab_rs


def lab2rgb(l_rs, ab_rs):
    l_cent = 50.
    l_norm = 100.
    ab_norm = 110.
    l = l_rs * l_norm + l_cent
    ab = ab_rs * ab_norm
    lab = torch.cat((l, ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))

    return out

def rgb2ycbcr(x):
    # Convert RGB to YCbCr
    weights_rgb_to_ycbcr = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                         [24.966, 112.0, -18.214]]).to(x.device)
    bias_rgb_to_ycbcr = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(x.device)
    x_ycbcr = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_ycbcr).permute(0, 3, 1, 2) + bias_rgb_to_ycbcr
    x_ycbcr = x_ycbcr / 255.
    y_channel = x_ycbcr[:, 0:1, :, :]
    uv_channel = x_ycbcr[:, 1:3, :, :]

    return x_ycbcr, y_channel, uv_channel


def ycbcr2rgb(x):
    x = x * 255.
    weights_ycbcr_to_rgb = 255. * torch.tensor([[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                                                [0.00625893, -0.00318811, 0]]).to(x)
    bias_ycbcr_to_rgb = torch.tensor([-222.921, 135.576, -276.836]).view(1, 3, 1, 1).to(x)
    x_rgb = torch.matmul(x.permute(0, 2, 3, 1), weights_ycbcr_to_rgb).permute(0, 3, 1, 2) \
            + bias_ycbcr_to_rgb
    x_rgb = x_rgb / 255.
    return x_rgb
