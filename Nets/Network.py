import math
import random

import pywt
import torch.fft
import torch
from einops import rearrange
from thop import profile, clever_format
from torch import nn, einsum
from torch.nn import functional as F

from Utilities.CUDA_Check import GPUorCPU

DEVICE = GPUorCPU.DEVICE


class SS_Block(nn.Module):
    def __init__(self):
        super(SS_Block, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        coeffs2 = pywt.dwt2(x.cpu().detach().numpy(), 'haar', mode='zero')  # 二维离散小波变换
        cA, (cH, cV, cD) = coeffs2  # cA:低频部分，cH:水平高频部分，cV:垂直高频部分，cD:对角线高频部分
        return cA, (cH, cV, cD)


class MSSE_Attention(nn.Module):
    def __init__(self, channels):
        super(MSSE_Attention, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        out = self.conv(x)
        out = F.mish(self.bn(out))
        out = torch.mean(out, dim=(2, 3))
        out = self.fc(out)
        out = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)
        out = out * x
        return out


class ParallelBlock(nn.Module):
    def __init__(self, channels):
        super(ParallelBlock, self).__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        # self.layer = nn.Conv2d(channels*3, channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out1 = F.mish(self.branch1(x))
        out2 = F.mish(self.branch2(x))
        out3 = F.mish(self.branch3(x))
        out = torch.cat((out1, out2, out3), dim=1)
        # out = self.layer(out)
        return out


class MSSE_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSSE_Module, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )
        self.attention = MSSE_Attention(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.parallel = ParallelBlock(out_channels)
        self.skip_connection = nn.Conv2d(in_channels, out_channels*3, kernel_size=1)
        self.channel_splitter = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        self.SS_Block = SS_Block()
        self.high_layer = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv(x)
        conn_x = out
        out = self.attention(out)
        out = out + residual
        out = self.parallel(out)
        out = out + self.skip_connection(x)
        out = self.channel_splitter(out)
        cA, (cH, cV, cD) = self.SS_Block(out)
        cA_tensor = torch.from_numpy(cA).to(x.device)
        cH_tensor = torch.from_numpy(cH).to(x.device)
        cV_tensor = torch.from_numpy(cV).to(x.device)
        cD_tensor = torch.from_numpy(cD).to(x.device)
        x_low = cA_tensor
        x_high = torch.cat([cH_tensor, cV_tensor, cD_tensor], dim=1)
        x_high = self.high_layer(x_high)
        return x_low, x_high, conn_x


class FSE_Module(nn.Module):
    def __init__(self, in_channels):
        super(FSE_Module, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels*2),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
        )
        self.SS_Block = SS_Block()
        self.high_layer = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.bottleneck(x)
        x = x + residual
        cA, (cH, cV, cD) = self.SS_Block(x)
        cA_tensor = torch.from_numpy(cA).to(x.device)
        cH_tensor = torch.from_numpy(cH).to(x.device)
        cV_tensor = torch.from_numpy(cV).to(x.device)
        cD_tensor = torch.from_numpy(cD).to(x.device)
        x_low = cA_tensor
        x_high = torch.cat([cH_tensor, cV_tensor, cD_tensor], dim=1)
        x_high = self.high_layer(x_high)
        return x_low, x_high


class RSE_Module(nn.Module):
    def __init__(self, in_channels):
        super(RSE_Module, self).__init__()
        # 第一条线路
        self.path1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.path2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.path3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=2)
        self.sub1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.Mish(inplace=True),
        )
        self.sub2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.Mish(inplace=True),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels*2, eps=1e-5),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.Mish(inplace=True),
        )
        self.SS_Block = SS_Block()
        self.high_layer = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        residual = x
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        sub1 = out1 - out2
        sub1 = self.sub1(sub1)
        sub1 = torch.cat([out1, sub1], dim=1)
        sub2 = out3 - out2
        sub2 = self.sub1(sub2)
        sub2 = torch.cat([out3, sub2], dim=1)
        x = self.layer(torch.cat([sub1, sub2], dim=1))
        x = x + residual
        cA, (cH, cV, cD) = self.SS_Block(x)
        cA_tensor = torch.from_numpy(cA).to(x.device)
        cH_tensor = torch.from_numpy(cH).to(x.device)
        cV_tensor = torch.from_numpy(cV).to(x.device)
        cD_tensor = torch.from_numpy(cD).to(x.device)
        x_low = cA_tensor
        x_high = torch.cat([cH_tensor, cV_tensor, cD_tensor], dim=1)
        x_high = self.high_layer(x_high)
        return x_low, x_high


class LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # 创建可供网络更新的张量
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, scale_factor, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      dilation=scale_factor, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, dim_head, scale_factor=None, dropout=0.):
        super().__init__()
        if scale_factor is None:
            scale_factor = [1, 2, 4]
        self.num_group = len(scale_factor)
        inner_dim = dim_head * self.num_group
        self.heads = self.num_group
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.Multi_scale_Token_Embeding = nn.ModuleList([])
        for i in range(len(scale_factor)):
            self.Multi_scale_Token_Embeding.append(nn.ModuleList([
                DepthWiseConv2d(dim, dim_head, proj_kernel, padding=scale_factor[i], stride=1,
                                scale_factor=scale_factor[i], bias=False),
                DepthWiseConv2d(dim, dim_head*2, proj_kernel, padding=scale_factor[i], stride=kv_proj_stride,
                                scale_factor=scale_factor[i], bias=False),
            ]))

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, d, h, w = x.shape
        Q, K, V = [], [], []
        for to_q, to_kv in self.Multi_scale_Token_Embeding:
            q = to_q(x)
            k, v = to_kv(x).chunk(2, dim=1)
            q, k, v = map(lambda t: rearrange(t, 'b d x y -> b (x y) d'), (q, k, v))
            Q.append(q)
            K.append(k)
            V.append(v)
        random.shuffle(Q)
        Q = torch.cat([Q[0], Q[1], Q[2]], dim=0)
        K = torch.cat([K[0], K[1], K[2]], dim=0)
        V = torch.cat([V[0], V[1], V[2]], dim=0)
        dots = einsum('b i d, b j d -> b i j', Q, K) * self.scale  # C = np.einsum('ij,jk->ik', A, B)保留想要的维度
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, V)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=self.num_group, x=h, y=w)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, dim_head, scale_factor, mlp_mult=4, dropout=0.):
        #dim=64, proj_kernel=3, kv_proj_stride=1, depth=2, scale_factor=[1, 2, 4], mlp_mult=8, dim_head=64, dropout=dropout
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                       dim_head=dim_head, scale_factor=scale_factor, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x   # PreNorm(1)
            x = ff(x) + x     # PreNorm(2)
        return x


class conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm(dim_out),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv3x3 = conv3x3(input_dim=dim_in, output_dim=dim_out)

    def forward(self, x, h, w):
        x = F.interpolate(x, mode='bilinear', size=(int(h), int(w)))
        x = self.conv3x3(x)
        return x


class Network(nn.Module):
    def __init__(self, img_channels=3, dropout=0.):
        super().__init__()

        self.Encoder1 = MSSE_Module(img_channels, 32)
        self.Encoder2_high = RSE_Module(32)
        self.Encoder2_low = FSE_Module(32)
        self.Encoder3_high = RSE_Module(48)
        self.Encoder3_low = FSE_Module(48)

        self.down1 = Downsample(64, 32, kernel_size=1, stride=1, padding=0)
        self.down2 = Downsample(64, 48, kernel_size=1, stride=1, padding=0)
        self.down3 = Downsample(96, 64, kernel_size=3, stride=2, padding=1)

        self.t0 = Transformer(dim=64, proj_kernel=3, kv_proj_stride=1, depth=2, scale_factor=[1, 2, 4], mlp_mult=8,
                              dim_head=64, dropout=dropout)
        self.tu1 = Transformer(dim=48, proj_kernel=3, kv_proj_stride=1, depth=1, scale_factor=[1, 2, 4], mlp_mult=4,
                               dim_head=48, dropout=dropout)
        self.tu2 = Transformer(dim=32, proj_kernel=3, kv_proj_stride=2, depth=1, scale_factor=[1, 2, 4], mlp_mult=4,
                               dim_head=32, dropout=dropout)

        self.up = Upsample(32, 32)
        self.up1 = Upsample(64, 48)
        self.up2 = Upsample(48, 32)
        self.up3 = Upsample(32, 32)

        self.skip_conn0 = conv3x3(input_dim=128, output_dim=32)
        self.skip_conn1 = conv3x3(input_dim=128, output_dim=32)
        self.skip_conn2 = conv3x3(input_dim=192, output_dim=48)

        self.Mixer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
        )
        self.Mixer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, stride=4, padding=3),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
        )
        self.Mixer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
        )

        self.ReconstructD = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, A, B):  # 3 224 224
        b, c, h, w = A.shape

        x_low1, x_high1, conn_x = self.Encoder1(A)
        y_low1, y_high1, conn_y = self.Encoder1(B)
        # skip_conn = self.skip_conn(torch.cat([conn_x, conn_y], dim=1))
        x = torch.cat([x_high1, y_high1], dim=1)
        y = torch.cat([x_low1, y_low1], dim=1)
        skip_conn0 = self.skip_conn0(torch.cat([x, y], dim=1))
        mix0 = torch.cat([x, y], dim=1)

        x, y = self.down1(x), self.down1(y)

        x_low2, x_high2 = self.Encoder2_high(x)
        y_low2, y_high2 = self.Encoder2_low(y)
        x = torch.cat([x_high2, y_high2], dim=1)
        y = torch.cat([x_low2, y_low2], dim=1)
        skip_conn1 = self.skip_conn1(torch.cat([x, y], dim=1))
        mix1 = torch.cat([x, y], dim=1)

        x, y = self.down2(x), self.down2(y)

        x_low3, x_high3 = self.Encoder3_high(x)
        y_low3, y_high3 = self.Encoder3_low(y)
        x = torch.cat([x_high3, y_high3], dim=1)
        y = torch.cat([x_low3, y_low3], dim=1)
        skip_conn2 = self.skip_conn2(torch.cat([x, y], dim=1))

        x, y = self.down3(x), self.down3(y)

        concatenation = torch.cat([x, y], dim=1)
        D = torch.cat([self.Mixer(concatenation),self.Mixer1(mix0),self.Mixer2(mix1)], dim=1)

        D = self.t0(D)
        D = self.up1(D, math.ceil(h / 8), math.ceil(w / 8))  # 48 28 28
        D = D + skip_conn2

        D = self.tu1(D)
        D = self.up2(D, math.ceil(h / 4), math.ceil(w / 4))
        D = D + skip_conn1

        D = self.tu2(D)
        D = self.up3(D, math.ceil(h / 2), math.ceil(w / 2))
        D = D + skip_conn0

        D = self.up(D, h, w)
        D = self.ReconstructD(D)

        return D


if __name__ == '__main__':
    test_tensor_A = torch.rand((1, 3, 520, 520)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 520, 520)).to(DEVICE)
    model = Network().to(DEVICE)
    model(test_tensor_A, test_tensor_B)
    print(model)
    flops, params = profile(model, inputs=(test_tensor_A, test_tensor_B))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {}, params: {}'.format(flops, params))
    Pre = model(test_tensor_A, test_tensor_B)
    print(Pre.shape)
