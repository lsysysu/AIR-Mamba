import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from pytorch_wavelets import DWTForward
from torch.autograd import Function
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchsummary import summary


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class HDWT(nn.Module):
    def __init__(self, in_channel, out_channel, use_input_lf=False):
        super().__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.use_input_lf = use_input_lf
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class GEBottleneck(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = (stride == 1 and inp == oup)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, inp, oup, expansion=2, stride=2):
        super().__init__()
        hidden_dim = int(inp * expansion)  # 通常 expansion=2 即可

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(),

            # 使用 1x1 conv 映射到输出维度
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        return self.conv(x)

class MobileMamba_blck(nn.Module):
    def __init__(self, sl, inc, d, outc):
        super().__init__()
        self.hidden_dim = d
        self.conv1 = conv_nxn_bn(inc, inc, 3)
        self.conv2 = conv_1x1_bn(inc, self.hidden_dim)
        self.mamba = Mamba(d_model=self.hidden_dim)
        self.conv3 = conv_1x1_bn(self.hidden_dim, outc)
        self.conv4 = conv_nxn_bn(outc, outc, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, D)
        x = self.mamba(x)
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class DynamicFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        proj_dim = 32  #
        self.proj_low = conv_1x1_bn(channels[0], proj_dim)
        self.proj_mid = conv_1x1_bn(channels[1], proj_dim)
        self.proj_high = conv_1x1_bn(channels[2], proj_dim)
        self.fused_channels = proj_dim * 3  # =48
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.fused_channels, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1))

    def forward(self, feat_low, feat_mid, feat_high):
        feat_low = F.adaptive_avg_pool2d(feat_low, feat_high.shape[-2:])
        feat_mid = F.adaptive_avg_pool2d(feat_mid, feat_high.shape[-2:])
        feat_low = self.proj_low(feat_low)
        feat_mid = self.proj_mid(feat_mid)
        feat_high = self.proj_high(feat_high)
        combined = torch.cat([feat_low, feat_mid, feat_high], dim=1)
        weights = self.weight_net(combined).unsqueeze(-1).unsqueeze(-1)
        fused = (weights[:, 0:1] * feat_low +
                 weights[:, 1:2] * feat_mid +
                 weights[:, 2:3] * feat_high)
        return torch.cat([feat_low, feat_mid, feat_high], dim=1)

class AirMamba(nn.Module):
    def __init__(self, image_size, channels, num_classes, expansion=4):
        super().__init__()
        ih, iw = image_size
        self.adaptive_gain = torch.jit.load("modules/adaptive_gain.pt")
        self.airm = nn.ModuleList([
            HDWT(channels[1], channels[2]),
            GEBottleneck(channels[2], channels[3], 1, expansion),
            GEBottleneck(channels[3], channels[4], 1, expansion),
            DownSample(channels[4], channels[5], expansion=expansion, stride=2),
            DownSample(channels[5], channels[6], expansion=expansion, stride=2),
            DownSample(channels[6], channels[7], expansion=expansion, stride=2)
        ])
        self.mamba = nn.ModuleList([
            MobileMamba_blck(sl=(ih // 8) ** 2, d=16, inc=channels[6], outc=channels[7]),
            MobileMamba_blck(sl=(ih // 16) ** 2,d=16, inc=channels[7], outc=channels[8]),
            MobileMamba_blck(sl=(ih // 32) ** 2,d=16, inc=channels[8], outc=channels[9])
        ])
        self.dynamic_fusion = DynamicFusion(channels=[channels[7], channels[8], channels[9]])
        self.conv2 = conv_1x1_bn(self.dynamic_fusion.fused_channels, channels[-1])
        self.adaptation_head = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.fc = nn.Linear(256, num_classes)
        self.pool = nn.AvgPool2d(ih // 32, 1)

    def forward(self, x, return_features=False):
        x = self.adaptive_gain(x)
        x = self.airm[0](x)
        x = self.airm[1](x)
        x = self.airm[2](x)
        x = self.airm[3](x)
        x = self.airm[4](x)
        feat_low = self.mamba[0](x)
        feat_mid = self.mamba[1](self.airm[5](feat_low))
        feat_high = self.mamba[2](feat_mid)
        x_fused = self.dynamic_fusion(feat_low, feat_mid, feat_high)
        x = self.conv2(x_fused)
        x = self.pool(x).view(x.size(0), -1)
        adapted_features = self.adaptation_head(x)
        return adapted_features if return_features else self.fc(adapted_features)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 42),
            nn.Dropout(0.7),
            nn.LeakyReLU(0.1),
            nn.Linear(42, 1)
        )
    def forward(self, x):
        return self.net(x)

def coral_loss(source, target):
    source_c = source - source.mean(0, keepdim=True)
    target_c = target - target.mean(0, keepdim=True)
    s_cov = torch.mm(source_c.T, source_c) / (source.size(0) - 1 + 1e-8)
    t_cov = torch.mm(target_c.T, target_c) / (target.size(0) - 1 + 1e-8)
    loss = torch.norm(s_cov - t_cov, p="fro")
    return loss


def airmamba_s():
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return AirMamba((256, 256), channels, num_classes=6)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model(model, input_size=(3, 256, 256), device='cuda'):
    model.eval().to(device)
    dummy_input = torch.randn(1, *input_size).to(device)

    # 1. Params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total Trainable Params: {total_params:,}")

    # 2.  FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"\n FLOPs (multiply-adds): {flops.total():,.0f}")
    print("\n  Per-Layer FLOP Breakdown:")
    print(flop_count_table(flops, max_depth=3))



if __name__ == '__main__':
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(5, 3, 256, 256).to(device)
    model = airmamba_s().to(device)
    out = model(dummy_input)
    print(out.shape)
    print("Params:", count_parameters(model))

