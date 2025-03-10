import torch
import torch.nn as nn
import torch.nn.functional as nnf
from layers import SpatialTransformer_block
from layers import ResizeTransformer_block
from layers import Conv_block
from layers import exploration
from layers import Encoder
from layers import evaluation
from layers import Def

from dcn.modules.deform_conv import DeformConv


class ESANet(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 channel_num: int = 8):
        super().__init__()

        self.encoder = Encoder(in_channels, channel_num)

        self.exploration4 = exploration(channel_num * 8 * 2, 24)
        self.exploration3 = exploration(channel_num * 4 * 2 + 24, 24)  # 16
        self.exploration2 = exploration(channel_num * 2 * 2 + 24, 24)  # 8
        self.exploration1 = exploration(channel_num * 1 * 2 + 24, 24)  # 8

        self.dcn4 = DeformConv(channel_num * 8, channel_num * 8, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])
        self.dcn3 = DeformConv(channel_num * 4, channel_num * 4, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])
        self.dcn2 = DeformConv(channel_num * 2, channel_num * 2, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])
        self.dcn1 = DeformConv(channel_num * 1, channel_num * 1, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])

        self.evaluation4 = evaluation(channel_num * 8 * 2, 24)    # 32
        self.evaluation3 = evaluation(channel_num * 4 * 2, 24)    # 16
        self.evaluation2 = evaluation(channel_num * 2 * 2, 24)    # 8
        self.evaluation1 = evaluation(channel_num * 1 * 2, 24)  # 8

        self.fine4 = nn.Sequential(Conv_block(24, 24), Def(24, 3))
        self.fine3 = nn.Sequential(Conv_block(24, 24), Def(24, 3))
        self.fine2 = nn.Sequential(Conv_block(24, 24), Def(24, 3))
        self.fine1 = nn.Sequential(Conv_block(24, 24), Def(24, 3))

        self.stn = SpatialTransformer_block()

        self.upsample = ResizeTransformer_block(resize_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        mov, fix = inputs[0], inputs[1]
        f1, f2, f3, f4 = self.encoder(fix)
        m1, m2, m3, m4 = self.encoder(mov)

        vector4 = self.exploration4(torch.cat([m4, f4], dim=1))
        f4_mov = self.dcn4(f4, vector4)
        f4_mov = nnf.pad(f4_mov, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w4 = self.sigmoid(self.evaluation4(torch.cat([f4_mov, m4], dim=1)))
        vector4 = w4 * vector4
        flow4 = self.fine4(vector4)
        flow4 = self.upsample(flow4)
        m3 = self.stn(m3, flow4)

        vector3 = self.exploration3(torch.cat([m3, f3, self.upsample(vector4)], dim=1))
        f3_mov = self.dcn3(f3, vector3)
        f3_mov = nnf.pad(f3_mov, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w3 = self.sigmoid(self.evaluation3(torch.cat([f3_mov, m3], dim=1)))
        vector3 = w3 * vector3
        flow3 = self.fine3(vector3)
        flow3 = self.upsample(flow3)
        m2 = self.stn(m2, flow3)

        vector2 = self.exploration2(torch.cat([m2, f2, self.upsample(vector3)], dim=1))
        f2_mov = self.dcn2(f2, vector2)
        f2_mov = nnf.pad(f2_mov, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w2 = self.sigmoid(self.evaluation2(torch.cat([f2_mov, m2], dim=1)))
        vector2 = w2 * vector2
        flow2 = self.fine2(vector2)
        flow2 = self.upsample(flow2)
        m1 = self.stn(m1, flow2)

        vector1 = self.exploration1(torch.cat([m1, f1, self.upsample(vector2)], dim=1))
        f1_mov = self.dcn1(f1, vector1)
        f1_mov = nnf.pad(f1_mov, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w1 = self.sigmoid(self.evaluation1(torch.cat([f1_mov, m1], dim=1)))
        vector1 = w1 * vector1
        flow1 = self.fine1(vector1)
        warp_mov = self.stn(mov, flow1)

        return warp_mov, flow1



