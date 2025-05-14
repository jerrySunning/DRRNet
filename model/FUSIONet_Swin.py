import torch.nn as nn
import torch
import torch.nn.functional as F
from model.swin_encoder import SwinTransformer
from model.DRRNet_modules import BasicConv2d, OCM, MDM, DRRM1, DRRM2, MMF, GRD


import os
class Network(nn.Module):
    # Swin Transformer based encoder decoder
    def __init__(self, channels=128):
        super(Network, self).__init__()
        self.encoder = SwinTransformer(img_size=384,
                                      embed_dim=128,
                                      depths=[2, 2, 18, 2],
                                      num_heads=[4, 8, 16, 32],
                                      window_size=12)
        pretrained_dict = torch.load('swin_base_patch4_window12_384_22k.pth')[ "model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)




        self.downPixelShuffle = torch.nn.PixelShuffle(2)

        self.reduce = nn.Sequential(
            BasicConv2d(channels * 2, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.up = nn.Sequential(
            BasicConv2d(channels // 4, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )

        self.Macro4 = OCM(1024, channels)
        self.Macro3 = OCM(512 + channels, channels)
        self.Macro2 = OCM(256 + channels, channels)
        self.Macro1 = OCM(128 + channels, channels)

        self.global_rough_de = GRD(1024 + channels, channels)
        self.feature_merge = MMF(channels)

        self.Micro4 = MDM(1024, channels)
        self.Micro3 = MDM(512 + channels, channels)
        self.Micro2 = MDM(256 + channels, channels)
        self.Micro1 = MDM(128 + channels, channels)

        self.drrm_1 = DRRM1(channels, channels)
        self.drrm_2 = DRRM2(channels, channels)



    def forward(self, x):
        image = x
        _, x4, x3, x2, x1 = self.encoder(x)
        g4 = self.Macro4(x4)
        g4_up = self.up(self.downPixelShuffle(g4))

        g3 = self.Macro3(torch.cat((x3, g4_up), 1))
        g3_up = self.up(self.downPixelShuffle(g3))

        g2 = self.Macro2(torch.cat((x2, g3_up), 1))
        g2_up = self.up(self.downPixelShuffle(g2))

        g1 = self.Macro1(torch.cat((x1, g2_up), 1))

        l4 = self.Micro4(x4)
        l4_up = self.up(self.downPixelShuffle(l4))

        l3 = self.Micro3(torch.cat((x3, l4_up), 1))
        l3_up = self.up(self.downPixelShuffle(l3))

        l2 = self.Micro2(torch.cat((x2, l3_up), 1))
        l2_up = self.up(self.downPixelShuffle(l2))
        l1 = self.Micro1(torch.cat((x1, l2_up), 1))

        p1 = self.global_rough_de(torch.cat((x4, self.feature_merge(g4, l4)), 1))

        x4 = self.drrm_1(g4, l4, p1)
        x3 = self.drrm_2(g3, l3, x4, p1)
        x2 = self.drrm_2(g2, l2, x3, x4)
        x1 = self.drrm_2(g1, l1, x2, x3)


        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f5 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)

        return f5, f4, f3, f2, f1

if __name__ == '__main__':
    net = Network(128)
    tensor = torch.rand(1, 3, 384, 384)
    out = net(tensor)