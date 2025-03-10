import torch
import torch.nn as nn
import torch.nn.functional as nnf


class Conv_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.Norm_1 = nn.InstanceNorm3d(out_channels)

    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.Norm_1(x)
        x_out = self.LeakyReLU(x)

        return x_out


class exploration(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.Conv_2 = nn.Conv3d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels//2)
        nn.init.zeros_(self.Conv_2.weight)
    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.norm(x)
        x = self.LeakyReLU(x)
        x = self.Conv_2(x)

        return x


class evaluation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.Norm_1 = nn.InstanceNorm3d(out_channels)
        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.Norm_2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.Norm_1(x)
        res = self.LeakyReLU(x)

        x = self.Conv_2(res)
        x = self.Norm_2(x)
        x = self.LeakyReLU(x) + res

        return x


class Flower(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.Conv_1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.Conv_2 = nn.Conv3d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.Conv_2(x)

        return x


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


class Def(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.Conv_1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.Conv_2 = nn.Conv3d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.Conv_2(x)

        return x


class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
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
        super().__init__()
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
