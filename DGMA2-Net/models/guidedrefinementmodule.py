import torch
import torch.nn as nn
import torch.nn.functional as F

# class GuidedRefinementModule(nn.Module):
#     def __init__(self, out_d):
#         super(GuidedRefinementModule, self).__init__()
#         self.out_d = out_d
#         self.conv_d5 = nn.Sequential(
#             nn.Conv2d(512, self.out_d[0], kernel_size=3, stride=1, padding=1),    #self.out_d
#             nn.BatchNorm2d(self.out_d[0]),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_d4 = nn.Sequential(
#             nn.Conv2d(256, self.out_d[1], kernel_size=3, stride=1, padding=1),     #self.out_d
#             nn.BatchNorm2d(self.out_d[1]),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_d3 = nn.Sequential(
#             nn.Conv2d(128, self.out_d[2], kernel_size=3, stride=1, padding=1),     #self.out_d
#             nn.BatchNorm2d(self.out_d[2]),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_d2 = nn.Sequential(
#             nn.Conv2d(64, self.out_d[3], kernel_size=3, stride=1, padding=1),      #self.out_d
#             nn.BatchNorm2d(self.out_d[3]),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
#         # feature refinement
#         d5 = self.conv_d5(d5_p + d5)
#         d4 = self.conv_d4(d4_p + d4)
#         d3 = self.conv_d3(d3_p + d3)
#         d2 = self.conv_d2(d2_p + d2)
#
#         return d5, d4, d3, d2




# #concat
class GuidedRefinementModule_concat(nn.Module):
    def __init__(self, out_d):
        super(GuidedRefinementModule_concat, self).__init__()
        self.out_d = out_d
        # self.conv_d5 = nn.Sequential(
        #     nn.Conv2d(512*2, self.out_d[0], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.out_d[0]),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_d4 = nn.Sequential(
        #     nn.Conv2d(256*2, self.out_d[1], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.out_d[1]),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_d3 = nn.Sequential(
        #     nn.Conv2d(128*2, self.out_d[2], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.out_d[2]),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_d2 = nn.Sequential(
        #     nn.Conv2d(64*2, self.out_d[3], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.out_d[3]),
        #     nn.ReLU(inplace=True)
        # )

        self.conv_d5 = nn.Sequential(
            nn.Conv2d(512 * 2, self.out_d[0], kernel_size=3, groups=self.out_d[0], stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[0]),
            nn.ReLU(inplace=True)
        )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(256 * 2, self.out_d[1], kernel_size=3, groups=self.out_d[1], stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[1]),
            nn.ReLU(inplace=True)
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(128 * 2, self.out_d[2], kernel_size=3, groups=self.out_d[2], stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[2]),
            nn.ReLU(inplace=True)
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(64 * 2, self.out_d[3], kernel_size=3, groups=self.out_d[3], stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[3]),
            nn.ReLU(inplace=True)
        )


    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        # feature refinement
        # concat
        # d5 = self.conv_d5(torch.cat([d5_p, d5], dim=1))
        # d4 = self.conv_d4(torch.cat([d4_p, d4], dim=1))
        # d3 = self.conv_d3(torch.cat([d3_p, d3], dim=1))
        # d2 = self.conv_d2(torch.cat([d2_p, d2], dim=1))

        # stack
        def stack(x1, x2):
            b, c, h, w = x1.size(0), x1.size(1), x1.size(2), x1.size(3)
            x_f = torch.stack((x1, x2), dim=2)
            x_f = torch.reshape(x_f, (b, -1, h, w))
            return x_f

        d5 = self.conv_d5(stack(d5_p, d5))
        d4 = self.conv_d4(stack(d4_p, d4))
        d3 = self.conv_d3(stack(d3_p, d3))
        d2 = self.conv_d2(stack(d2_p, d2))

        return d5, d4, d3, d2