import torch
import torch.nn as nn
import torch.nn.functional as nnf

class Encoder(nn.Module):
    def __init__(self, in_channels:int, channel_num:int):
        super(Encoder, self).__init__()
        self.d_init = Conv_block(in_channels, channel_num)  # 8
        self.d_e1 = Conv_block(channel_num, channel_num * 2)  # 16
        self.d_e2 = Conv_block(channel_num * 2, channel_num * 4)  # 32
        self.d_e3 = Conv_block(channel_num * 4, channel_num * 8)  # 64
        # 非参数版
        self.downsample = nn.AvgPool3d(2, stride=2)
    def forward(self, dvol):
        x_in = self.d_init(dvol)
        x = self.downsample(x_in)
        x_1 = self.d_e1(x)
        x = self.downsample(x_1)
        x_2 = self.d_e2(x)
        x = self.downsample(x_2)
        x_3 = self.d_e3(x)
        return x_in, x_1, x_2, x_3

class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super(SpatialTransformer_block, self).__init__()
        self.mode = mode
    def forward(self, src, flow):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super(ResizeTransformer_block, self).__init__()
        self.factor = resize_factor
        self.mode = mode
    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=2, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv3d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.InstanceNorm3d(gate_channel//reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.LeakyReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d'%i, nn.Conv3d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d'%i, nn.InstanceNorm3d(gate_channel//reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d'%i, nn.LeakyReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv3d(gate_channel//reduction_ratio, 1, kernel_size=1))
    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

class ChannelGate1(nn.Module):
    '''
    基于全通道间交互的注意力
    '''
    def __init__(self, k_size=3):
        super(ChannelGate1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act_func = nn.LeakyReLU()
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), 1, -1)
        y = self.conv(y)
        y = self.act_func(y)
        y = y.view(x.size(0), x.size(1), 1, 1, 1)
        return y.expand_as(x)

class ChannelGate2(nn.Module):
    '''
    基于每个通道自身的注意力
    '''
    def __init__(self, channel, k_size):
        super(ChannelGate2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              groups=channel, bias=False)
        self.act_func = nn.LeakyReLU()
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.unsqueeze(-1)
        y = self.conv(y)
        y = self.act_func(y)
        y = y.view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Exploration(nn.Module):
    '''
    channel_attention is inspired from "https://github.com/BangguWu/ECANet/tree/master/models"
    spatial_attention is inspired from "https://github.com/Jongchan/attention-module/blob/master/MODELS"
    the entire structure is following "https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py"
    '''
    def __init__(self, eca_version_1: bool, channel: int, reduction_ratio: int):
        super(Exploration, self).__init__()
        if eca_version_1: # 关注基于全通道间交互的注意力，推荐
            self.channel_att = ChannelGate1()
        else:             # 关注基于每个通道自身的注意力
            self.channel_att = ChannelGate2(channel=channel, k_size=3)
        self.spatial_att = SpatialGate(gate_channel=channel, reduction_ratio=reduction_ratio, dilation_conv_num=2, dilation_val=4)
        self.conv = nn.Conv3d(channel, 24, 3, 1, 1)
    def forward(self, in_tensor):
        att = 1 + nnf.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return self.conv(att * in_tensor)

class Evaluation(nn.Module):
    def __init__(self, in_channels, channels):
        super(Evaluation, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, channels, 3, 1, 1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Conv3d(channels, 8, 3, 1, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, in_tensor):
        x = self.conv1(in_tensor)
        shortcut = x
        x = self.conv2(x)
        x = self.conv3(x + shortcut)
        return self.softmax(x)

def Refinement(W, offset_field):
    _, c, _, _, _ = offset_field.shape
    num_fields = c // 3
    weighted_field = 0
    for i in range(num_fields):
        w = offset_field[:, 3*i: 3*(i+1)]
        weight_map = W[:, i: (i+1)]
        weighted_field = weighted_field + w * weight_map
    return weighted_field

class Deformation_Estimator(nn.Module):
    def __init__(self, in_channels):
        super(Deformation_Estimator, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels * 2, 3, 1, 1)
        self.reg_head = nn.Conv3d(in_channels * 2, 3, kernel_size=3, stride=1, padding=1)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    def forward(self, in_tensor):
        return self.reg_head(self.conv(in_tensor))