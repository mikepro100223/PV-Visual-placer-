"""Small full-resolution U-Net for surveyed rooftop-superstructure masks."""
import torch
from torch import nn
import torch.nn.functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(4, outputs), nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(4, outputs), nn.SiLU())


class ObstacleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = block(3,16), block(16,32), block(32,64)
        self.bridge = block(64,128)
        self.dec3, self.dec2, self.dec1 = block(192,64), block(96,32), block(48,16)
        self.head = nn.Conv2d(16,1,1)

    def forward(self, image):
        a = self.enc1(image)
        b = self.enc2(F.max_pool2d(a,2))
        c = self.enc3(F.max_pool2d(b,2))
        d = self.bridge(F.max_pool2d(c,2))
        d = self.dec3(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        d = self.dec2(torch.cat([F.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.head(self.dec1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))
