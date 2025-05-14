import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from model.GatedConv import GatedConv2dWithActivation
from einops import rearrange
import numbers

from timm.models.layers import to_2tuple

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError

def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)



class MultiScaleFFN(nn.Module):
    def __init__(self, channels, expansion_factor, use_bias):
        super(MultiScaleFFN, self).__init__()
        hidden_dim = int(channels * expansion_factor)
        self.input_proj = nn.Conv2d(channels, hidden_dim * 2, kernel_size=1, bias=use_bias)
        self.dwconv_3x3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_dim * 2, bias=use_bias)
        self.dwconv_5x5 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_dim * 2, bias=use_bias)
        self.dwconv_7x7 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_dim * 2, bias=use_bias)

        self.output_proj = nn.Conv2d(hidden_dim * 3, channels, kernel_size=1, bias=use_bias)

    def forward(self, x):
        x = self.input_proj(x)
        x1_3, x2_3 = self.dwconv_3x3(x).chunk(2, dim=1)
        x_3x3 = F.gelu(x1_3) * x2_3
        x1_5, x2_5 = self.dwconv_5x5(x).chunk(2, dim=1)
        x_5x5 = F.gelu(x1_5) * x2_5
        x1_7, x2_7 = self.dwconv_7x7(x).chunk(2, dim=1)
        x_7x7 = F.gelu(x1_7) * x2_7

        x = self.output_proj(torch.cat((x_3x3, x_5x5, x_7x7), 1))
        return x

    def initialize(self):
        weight_init(self)



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.qkv1conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv2conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv3conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_3 = self.qkv1conv_3(self.qkv_0(x))
        k_3 = self.qkv2conv_3(self.qkv_1(x))
        v_3 = self.qkv3conv_3(self.qkv_2(x))

        q_3 = rearrange(q_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_3 = rearrange(k_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_3 = rearrange(v_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_3 = torch.nn.functional.normalize(q_3, dim=-1)
        k_3 = torch.nn.functional.normalize(k_3, dim=-1)
        attn_3 = (q_3 @ k_3.transpose(-2, -1)) * self.temperature
        attn_3 = attn_3.softmax(dim=-1)
        out_3 = (attn_3 @ v_3)
        out_3 = rearrange(out_3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_5 = self.qkv1conv_5(self.qkv_0(x))
        k_5 = self.qkv2conv_5(self.qkv_1(x))
        v_5 = self.qkv3conv_5(self.qkv_2(x))

        q_5 = rearrange(q_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_5 = rearrange(k_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_5 = rearrange(v_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_5 = torch.nn.functional.normalize(q_5, dim=-1)
        k_5 = torch.nn.functional.normalize(k_5, dim=-1)
        attn_5 = (q_5 @ k_5.transpose(-2, -1)) * self.temperature
        attn_5 = attn_5.softmax(dim=-1)
        out_5 = (attn_5 @ v_5)
        out_5 = rearrange(out_5, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_7 = self.qkv1conv_7(self.qkv_0(x))
        k_7 = self.qkv2conv_7(self.qkv_1(x))
        v_7 = self.qkv3conv_7(self.qkv_2(x))

        q_7 = rearrange(q_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_7 = rearrange(k_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_7 = rearrange(v_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_7 = torch.nn.functional.normalize(q_7, dim=-1)
        k_7 = torch.nn.functional.normalize(k_7, dim=-1)
        attn_7 = (q_7 @ k_7.transpose(-2, -1)) * self.temperature
        attn_7 = attn_7.softmax(dim=-1)
        out_7 = (attn_7 @ v_7)
        out_7 = rearrange(out_7, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat((out_3,out_5,out_7),1))
        return out



    def initialize(self):
        weight_init(self)




class MSA_head(nn.Module):  # Multi-scale transformer block
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MultiScaleFFN(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


# OmniContext Module (MDM)
class OCM(nn.Module):
    def __init__(self, in_channel, out_channel, num_groups=4):
        super().__init__()
        self.scale_branch = nn.ModuleDict({
            'l': nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                nn.BatchNorm2d(out_channel)
            ),
            'm': nn.Conv2d(in_channel, out_channel, 1),
            's': nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channel, out_channel, 3, 1, 1)
            )
        })

        self.intra_conv = nn.ModuleDict({
            'l': ConvBNReLU(out_channel, out_channel, 3, 1, 1),
            'm': ConvBNReLU(out_channel, out_channel, 3, 1, 1),
            's': ConvBNReLU(out_channel, out_channel, 3, 1, 1)
        })

        self.inter_fusion = nn.Sequential(
            nn.Conv2d(3 * out_channel, 3 * out_channel, 1),
            nn.BatchNorm2d(3 * out_channel),
            nn.ReLU(True)
        )

        self.attn = nn.Sequential(
            nn.Conv2d(3 * out_channel, out_channel // num_groups, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channel // num_groups, 3, 1),
            nn.Softmax(dim=1)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        scales = {
            'l': self.scale_branch['l'](x),  # (b, c, h/2, w/2)
            'm': self.scale_branch['m'](x),  # (b, c, h, w)
            's': self.scale_branch['s'](x)  # (b, c, 2h, 2w)
        }

        for k in scales:
            if k != 'm':
                scales[k] = F.interpolate(scales[k], size=(h, w), mode='bilinear')

        intra_feats = {
            k: self.intra_conv[k](v)
            for k, v in scales.items()
        }

        lms = torch.cat(list(intra_feats.values()), dim=1)  # (b, 3c, h, w)
        fused = self.inter_fusion(lms)  # 3c→3c
        attn_weights = self.attn(fused).unsqueeze(2)  # (b, 3, 1, h, w)

        weighted = torch.stack(list(intra_feats.values()), dim=1)  # (b, 3, c, h, w)
        global_feat = (attn_weights * weighted).sum(dim=1)  # (b, c, h, w)

        residual = self.residual(x)
        return global_feat + residual

# GlobalRoughDecoder (GRD)
class GRD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GRD, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=1,dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3, 1, 1),nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channel, out_channel//2, 3, padding=1),nn.BatchNorm2d(out_channel//2),nn.PReLU(), nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channel//2, 1, 1)
        )

        self.msa_head = MSA_head(dim=out_channel)



    def forward(self, x):

        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x_multi = self.msa_head(x1)
        x_res  = self.res(x)
        x = self.reduce(torch.cat((x_res,x_multi),1)) + x0
        x = self.out(x)
        return x

# “MicroDetail Module (MDM)”
class MDM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MDM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ) for rate in [1, 3, 5, 7]
        ])
        self.reduce_aspp = nn.Sequential(
            nn.Conv2d(out_channel * 4, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.reduce_dw = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.se_fusion = SELayer(out_channel * 2)
        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, 3, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        aspp_feats = [branch(x1) for branch in self.aspp]
        aspp_out = self.reduce_aspp(torch.cat(aspp_feats, dim=1))
        dw3 = self.dw_conv3(x1)
        dw5 = self.dw_conv5(x1)
        dw7 = self.dw_conv7(x1)
        dw_out = self.reduce_dw(torch.cat([dw3, dw5, dw7], dim=1))
        local_feat = self.se_fusion(torch.cat([aspp_out, dw_out], dim=1))
        res_feat = self.res(x)
        out = self.fuse_conv(torch.cat([local_feat, res_feat], dim=1)) + x0
        return out



class GroupFusionBlock(nn.Module):


    def __init__(self, channels):
        super(GroupFusionBlock, self).__init__()
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.freq_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        spatial_out = self.conv_spatial(x)

        x_fft = torch.fft.fft2(x, norm="ortho")  # [B, C, H, W]
        x_fft_flat = x_fft.view(B, C, -1)
        amp = torch.abs(x_fft_flat)  #  [B, C, H*W]
        amp_trans = amp.transpose(1, 2)  # [B, H*W, C]
        attn = self.freq_mlp(amp_trans)  # [B, H*W, C]
        attn = torch.sigmoid(attn)
        attn = attn.transpose(1, 2).view(B, C, H, W)
        x_fft_mod = x_fft * attn.type_as(x_fft)
        freq_out = torch.fft.ifft2(x_fft_mod, norm="ortho").real

        out = spatial_out + self.gamma * freq_out
        return out

# "Macro-Micro Fusion"（MMF）
class MMF(nn.Module):
    def __init__(self, in_channels=128, groups=4):
        super(MMF, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        group_channels = in_channels // groups

        self.pre_conv_G = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pre_conv_L = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.group_fusion = nn.ModuleList([
            GroupFusionBlock(2 * group_channels) for _ in range(groups)
        ])

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gated_conv = GatedConv2dWithActivation(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
            activation=nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, G, L):
        G_feat = self.pre_conv_G(G)
        L_feat = self.pre_conv_L(L)

        fusion_input = torch.cat([G_feat, L_feat], dim=1)
        groups = fusion_input.chunk(self.groups, dim=1)
        fused_groups = [self.group_fusion[i](groups[i]) for i in range(self.groups)]
        fused = torch.cat(fused_groups, dim=1)

        concat_feat = torch.cat([fusion_input, fused], dim=1)
        out = self.reduce(concat_feat)
        out = self.gated_conv(out) * out + out
        out = self.final_conv(out)
        return out


# DRRM – Dual Reverse Refinement Module
class DRRM1(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(DRRM1, self).__init__()
        self.GL_FI = MMF(in_channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.freq_weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.merge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.channel_attn = SELayer(in_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

    def forward(self, G, L, prior_cam):
        fused_feat = self.GL_FI(G, L)
        prior_resized = F.interpolate(prior_cam, size=fused_feat.size()[2:], mode='bilinear', align_corners=True)
        prior_expanded = prior_resized.expand(-1, fused_feat.size(1), -1, -1)

        init_fuse = self.fusion_conv(torch.cat([fused_feat, prior_expanded], dim=1))

        spatial_out = self.spatial_branch(init_fuse)
        fft_feat = torch.fft.fft2(init_fuse, norm="ortho")
        fft_real = fft_feat.real
        freq_feat = self.freq_conv(fft_real)
        weight_map = self.freq_weight(freq_feat)
        modulated_fft = fft_feat * weight_map
        freq_out = torch.fft.ifft2(modulated_fft, norm="ortho").real

        branch_merge = self.merge_conv(torch.cat([spatial_out, freq_out], dim=1))
        branch_merge = self.channel_attn(branch_merge)

        combined_final = torch.cat([branch_merge, fused_feat], dim=1)
        out = self.out_conv(combined_final)
        out = out + prior_resized

        return out



# Dual Reverse Refinement Module, DRRM
class DRRM2(nn.Module):


    def __init__(self, in_channels, mid_channels):
        super(DRRM2, self).__init__()
        self.GL_FI = MMF(in_channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.freq_weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.merge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_attn = SELayer(in_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

    def forward(self, G, L, x1, prior_cam):
        fused_feat = self.GL_FI(G, L)
        prior_resized = F.interpolate(prior_cam, size=fused_feat.size()[2:], mode='bilinear', align_corners=True)
        x1_resized = F.interpolate(x1, size=fused_feat.size()[2:], mode='bilinear', align_corners=True)

        init_concat = torch.cat([
            fused_feat,
            prior_resized.expand(-1, fused_feat.size(1), -1, -1),
            x1_resized.expand(-1, fused_feat.size(1), -1, -1)
        ], dim=1)
        init_fuse = self.fusion_conv(init_concat)

        spatial_out = self.spatial_branch(init_fuse)
        fft_feat = torch.fft.fft2(init_fuse, norm="ortho")
        fft_real = fft_feat.real
        freq_feat = self.freq_conv(fft_real)
        weight_map = self.freq_weight(freq_feat)
        modulated_fft = fft_feat * weight_map
        freq_out = torch.fft.ifft2(modulated_fft, norm="ortho").real

        branch_merge = self.merge_conv(torch.cat([spatial_out, freq_out], dim=1))
        branch_merge = self.channel_attn(branch_merge)

        prior_res = -1 * torch.sigmoid(prior_resized) + 1
        x1_res = -1 * torch.sigmoid(x1_resized) + 1
        prior_weight = prior_res + x1_res
        weighted_res = prior_weight.expand(-1, fused_feat.size(1), -1, -1) * fused_feat

        combined = torch.cat([branch_merge, weighted_res], dim=1)
        out = self.out_conv(combined)
        out = out + prior_resized + x1_resized
        return out






