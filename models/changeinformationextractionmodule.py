import torch
import torch.nn as nn
import torch.nn.functional as F

class Refine(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Refine, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.inchannel + self.outchannel, self.outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)  #512
        x1 = self.conv1(x1)
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv2(x_f)
        return x_f

class ChangeInformationExtractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeInformationExtractionModule, self).__init__()

        self.in_d = in_d
        self.out_d = out_d

        self.conv_dr = nn.Sequential(
            nn.Conv2d(64, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.refine1 = Refine(512, 256)
        self.refine2 = Refine(256, 128)
        self.refine3 = Refine(128, 64)
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),  # 0 0
            nn.Conv2d(self.in_d, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),  # 1 1
            nn.Conv2d(self.in_d, 256, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),  # 2 2
            nn.Conv2d(self.in_d, 512, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def forward(self, d5, d4, d3, d2):

        # refine
        r1 = self.refine1(d5, d4)
        r2 = self.refine2(r1, d3)
        x = self.refine3(r2, d2)

        x = self.conv_dr(x)

        # pooling
        p2 = x
        p3 = self.conv_pool1(x)
        p4 = self.conv_pool2(x)
        p5 = self.conv_pool3(x)

        return p5, p4, p3, p2


if __name__ =='__main__':
    x = torch.randn((4, 64, 64, 64))
    out = model(x)
    print(out.shape)
