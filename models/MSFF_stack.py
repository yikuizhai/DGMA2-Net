import torch
import torch.nn as nn

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.convmix = nn.Sequential(nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)

        return out

if __name__ == '__main__':
    x = torch.randn((32, 256, 32, 32))
    model = MPFL(256,64)
    out = model(x)
    print(out.shape)