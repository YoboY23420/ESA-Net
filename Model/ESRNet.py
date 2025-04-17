import torch
import torch.nn as nn
import torch.nn.functional as nnf
from layers import SpatialTransformer_block
from layers import ResizeTransformer_block
from layers import Encoder
from layers import Exploration, Evaluation, Deformation_Estimator, Refinement
from dcn.modules.deform_conv import DeformConv

class ESR_Net(nn.Module):
    def __init__(self, in_channels: int = 1, channel_num: int = 8):
        super(ESR_Net, self).__init__()

        self.encoder = Encoder3(in_channels, channel_num)

        self.exploration4 = Exploration(True, channel_num * 8 * 2, reduction_ratio=4)
        self.exploration3 = Exploration(True, channel_num * 4 * 2 + 3, reduction_ratio=4)  # 16
        self.exploration2 = Exploration(True, channel_num * 2 * 2 + 3, reduction_ratio=4)  # 8
        self.exploration1 = Exploration(True, channel_num * 1 * 2 + 3, reduction_ratio=4)  # 8

        self.dcn4 = DeformConv(channel_num * 8, channel_num * 8, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])
        self.dcn3 = DeformConv(channel_num * 4, channel_num * 4, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])
        self.dcn2 = DeformConv(channel_num * 2, channel_num * 2, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])
        self.dcn1 = DeformConv(channel_num * 1, channel_num * 1, kernel_size=[2, 2, 2], stride=[1, 1, 1],
                               padding=[0, 0, 0])

        self.test4 = Evaluation(channel_num * 8 * 2, 24)    # 32
        self.test3 = Evaluation(channel_num * 4 * 2, 24)    # 16
        self.test2 = Evaluation(channel_num * 2 * 2, 24)    # 8
        self.test1 = Evaluation(channel_num * 1 * 2, 24)    # 8

        self.fine4 = Deformation_Estimator(3)
        self.fine3 = Deformation_Estimator(3)
        self.fine2 = Deformation_Estimator(3)
        self.fine1 = Deformation_Estimator(3)

        self.stn = SpatialTransformer_block()

        self.upsample = ResizeTransformer_block(resize_factor=2)

    def forward(self, inputs):
        mov, fix = inputs[0], inputs[1]
        f1, f2, f3, f4 = self.encoder(fix)
        m1, m2, m3, m4 = self.encoder(mov)

        vector4 = self.exploration4(torch.cat([m4, f4], dim=1))
        m4_fix = self.dcn4(m4, vector4)
        m4_fix = nnf.pad(m4_fix, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w4 = self.test4(torch.cat([m4_fix, f4], dim=1))
        vector4 = Refinement(w4, vector4)
        flow4 = self.fine4(vector4)
        flow4 = self.upsample(flow4)
        m3 = self.stn(m3, flow4)

        vector3 = self.exploration3(torch.cat([m3, f3, self.upsample(vector4)], dim=1))
        m3_fix = self.dcn3(m3, vector3)
        m3_fix = nnf.pad(m3_fix, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w3 = self.test3(torch.cat([m3_fix, f3], dim=1))
        vector3 =Refinement(w3, vector3)
        flow3 = self.fine3(vector3)
        flow3 = self.upsample(flow3)
        m2 = self.stn(m2, flow3)

        vector2 = self.exploration2(torch.cat([m2, f2, self.upsample(vector3)], dim=1))
        m2_fix = self.dcn2(m2, vector2)
        m2_fix = nnf.pad(m2_fix, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w2 = self.test2(torch.cat([m2_fix, f2], dim=1))
        vector2 =Refinement(w2, vector2)
        flow2 = self.fine2(vector2)
        flow2 = self.upsample(flow2)
        m1 = self.stn(m1, flow2)

        vector1 = self.exploration1(torch.cat([m1, f1, self.upsample(vector2)], dim=1))
        m1_fix = self.dcn1(m1, vector1)
        m1_fix = nnf.pad(m1_fix, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
        w1 = self.test1(torch.cat([m1_fix, f1], dim=1))
        vector1 = Refinement(w1, vector1)
        flow1 = self.fine1(vector1)
        warp_mov = self.stn(mov, flow1)

        return warp_mov, flow1


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    model = ESR_Net().cuda()
    # model.eval()
    in_tensor = torch.Tensor(1,1,160,192, 160).cuda()
    in_tensor = [in_tensor, in_tensor]
    out_tensor = model(in_tensor)
    params = count_parameters(model)
    print(f"参数量: {params}")