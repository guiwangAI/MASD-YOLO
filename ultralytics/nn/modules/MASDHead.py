import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from ultralytics.utils.checks import check_version

__all__ = ['MASDHead']

TORCH_1_10 = check_version(torch.__version__, '1.10.0')

MASD_NUM_BLOCKS = 2


class DeformConv(nn.Module):
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1):
        super(DeformConv, self).__init__()
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2 * kernel_size[0] * kernel_size[1],
                      kernel_size, padding=padding, stride=stride, dilation=dilation, bias=True)
        )
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False
        )

    def forward(self, x):
        offsets = self.offset_net(x)
        return self.deform_conv(x, offsets)


class AgileConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class deformable_agile_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AgileConv(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class MASD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_MASD=True):
        super().__init__()
        self.nonlinear = nn.SiLU()
        self.use_MASD = use_MASD
        padding = kernel_size // 2

        self.base_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.depth_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.dilation_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.branch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        if use_MASD:
            self.masd = deformable_agile_Attention(out_channels)

    def forward(self, x):
        if self.training:
            base_out = self.base_branch(x)
            depth_out = self.depth_branch(x)
            dilation_out = self.dilation_branch(x)

            attn = self.branch_attn(x)
            w1, w2, w3 = attn[:, 0:1], attn[:, 1:2], attn[:, 2:3]
            out = w1 * base_out + w2 * depth_out + w3 * dilation_out

            if self.use_MASD:
                out = self.masd(out)
            return self.nonlinear(out)
        else:
            with torch.no_grad():
                base_fused = self._forward_fused_branch(x, self.base_branch)
                depth_fused = self._forward_fused_branch(x, self.depth_branch)
                dilation_fused = self._forward_fused_branch(x, self.dilation_branch)

                attn = self.branch_attn(x)
                w1, w2, w3 = attn[:, 0:1], attn[:, 1:2], attn[:, 2:3]

            out = w1 * base_fused + w2 * depth_fused + w3 * dilation_fused
            if self.use_MASD:
                out = self.masd(out)
            return self.nonlinear(out)

    def _forward_fused_branch(self, x, branch):
        conv, bn = branch[0], branch[1]
        w, b = self._get_fused_conv_bn(conv, bn)
        return F.conv2d(x, w, b, stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups)

    def _get_fused_conv_bn(self, conv, bn):
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        std = torch.sqrt(var + bn.eps)
        w = conv.weight * (gamma / std).reshape(-1, 1, 1, 1)
        b = beta - (mean * gamma) / std
        return w, b


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class MASDHead(nn.Module):
    dynamic = True
    export = False
    shape = True
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        global MASD_NUM_BLOCKS
        self.num_blocks = MASD_NUM_BLOCKS

        self.DBB = nn.ModuleList([
            nn.Sequential(*[
                MASD(x, x, 3, use_MASD=True)
                for _ in range(self.num_blocks)
            ]) for x in ch
        ])

        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape

        for i in range(self.nl):
            x[i] = self.DBB[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a.bias.data[:] = 1.0
            b.bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)