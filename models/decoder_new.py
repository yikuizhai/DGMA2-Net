import torch
import torch.nn as nn
import torch.nn.functional as F
from models.guidedrefinementmodule import GuidedRefinementModule_concat
from models.changeinformationextractionmodule import ChangeInformationExtractionModule
from models.BAM import DEAM

class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=False)
        self.DEAM1 = DEAM(512)
        self.DEAM2 = DEAM(256)
        self.DEAM3 = DEAM(128)
        self.DEAM4 = DEAM(64)

        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

    def forward(self, d5, d4, d3, d2, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        x1_5, x2_5 = self.DEAM1(x1_5, d5), self.DEAM1(x2_5, d5)
        x1_4, x2_4 = self.DEAM2(x1_4, d4), self.DEAM2(x2_4, d4)
        x1_3, x2_3 = self.DEAM3(x1_3, d3), self.DEAM3(x2_3, d3)
        x1_2, x2_2 = self.DEAM4(x1_2, d2), self.DEAM4(x2_2, d2)

        d5 = x1_5 + x2_5 + d5
        d4 = x1_4 + x2_4 + d4
        d3 = x1_3 + x2_3 + d3
        d2 = x1_2 + x2_2 + d2

        d5 = self.conv1(d5)
        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        d4 = self.conv2(d4 + d5)
        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)
        d3 = self.conv3(d3 + d4)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.conv4(d2 + d3)

        mask = self.cls(d2)

        return mask


if __name__ == '__main__':
    x0 = torch.randn((32, 3, 256, 256)).cuda()
    x1 = torch.randn((32, 64, 64, 64)).cuda()
    x2 = torch.randn((32, 128, 32, 32)).cuda()
    x3 = torch.randn((32, 256, 16, 16)).cuda()
    x4 = torch.randn((32, 512, 8, 8)).cuda()
    model = ChangeInformationExtractionModule(64, 64).cuda()
    aa = GuidedRefinementModule_concat(64).cuda()
    d5, d4, d3, d2 = model(x4, x3, x2, x1)
    d5, d4, d3, d2 = aa(d5, d4, d3, d2, x4, x3, x2, x1)
    dd = Decoder(64, 1).cuda()
    out = dd(d5, d4, d3, d2)
    mask = F.interpolate(out, x0.size()[2:], mode='bilinear', align_corners=True)
    mask = torch.sigmoid(mask)
    print(mask.shape)

