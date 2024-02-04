import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
from models.mdfm import MDFM
from models.decoder_new import ChangeInformationExtractionModule, GuidedRefinementModule_concat, Decoder


class DGMAANet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(DGMAANet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.mid_d = 64
        self.MDFM5 = MDFM(512, 512)            #512
        self.MDFM4 = MDFM(256, 256)            #256
        self.MDFM3 = MDFM(128, 128)            #128
        self.MDFM2 = MDFM(64, 64)              #64

        ##DAM1
        self.CIEM1 = ChangeInformationExtractionModule(64, 64)
        self.GRM1 = GuidedRefinementModule_concat(out_d=[512, 256, 128, 64])   ######################[512, 256, 128, 64]

        ##DAM2
        self.CIEM2 = ChangeInformationExtractionModule(64, 64)
        self.GRM2 = GuidedRefinementModule_concat(out_d=[512, 256, 128, 64])   ###################

        self.decoder = Decoder(self.mid_d, output_nc)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        d5 = self.MDFM5(x1_5, x2_5)  # 1/32
        d4 = self.MDFM4(x1_4, x2_4)  # 1/16
        d3 = self.MDFM3(x1_3, x2_3)  # 1/8
        d2 = self.MDFM2(x1_2, x2_2)  # 1/4

        ##DAM1
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        ##DAM2
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        mask = self.decoder(d5, d4, d3, d2, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.sigmoid(mask)

        return mask

if __name__ == '__main__':
    x1 = torch.randn((4, 3, 256, 256)).cuda()
    x2 = torch.randn((4, 3, 256, 256)).cuda()
    model = BaseNet(3, 1).cuda()
    out = model(x1, x2)
    print(out.shape)
