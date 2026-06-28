import torch
import torch.nn as nn

__all__ = ['ReLConv']

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def fuse_conv_bn(conv, bn):
    with torch.no_grad():
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros_like(mean)
        
        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = (b - mean) * (gamma / std) + beta
        
        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            groups=conv.groups,
            dilation=conv.dilation,
            bias=True
        )
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        return fused_conv

class Down(nn.Module):
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.stride = stride
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, stride, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, self.stride, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, self.stride, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class RepConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c1, self.c2, self.k = c1, c2, k
        self.s, self.g, self.d = s, g, d

        self.conv_main = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn_main = nn.BatchNorm2d(c2)

        self.has_1x1_branch = (k > 1) and (g == 1 or g == c2)
        if self.has_1x1_branch:
            self.conv_1x1 = nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self._fused = False

    def train(self, mode=True):
        super().train(mode)
        if not mode and not self._fused:
            self.fuse()
        return self

    def forward(self, x):
        if self._fused:
            return self.forward_fuse(x)
        out = self.bn_main(self.conv_main(x))
        if self.has_1x1_branch:
            out += self.bn_1x1(self.conv_1x1(x))
        return self.act(out)

    def forward_fuse(self, x):
        return self.act(self.conv_main(x))

    def fuse(self):
        if self._fused:
            return
            
        self.conv_main = fuse_conv_bn(self.conv_main, self.bn_main)
        self.bn_main = nn.Identity()

        if self.has_1x1_branch:
            conv_1x1_fused = fuse_conv_bn(self.conv_1x1, self.bn_1x1)
            pad = (self.k - 1) // 2
            self.conv_main.weight.data += torch.nn.functional.pad(conv_1x1_fused.weight, (pad, pad, pad, pad))
            self.conv_main.bias.data += conv_1x1_fused.bias.data
            self.has_1x1_branch = False
            del self.conv_1x1
            del self.bn_1x1
        
        self._fused = True

class ReLConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, c3=None):
        super().__init__()
        if c3 is None:
            c3 = c1
        if c3 % 2 != 0:
            c3 += 1

        self.cv1 = RepConv(c1, c3, k=1, s=1, act=True)
        self.cv2 = RepConv(c3, c3, k=k, s=1, g=c3, act=False)
        self.cv3 = Down(c3, c2, stride=s)

    def forward(self, x):
        main = self.cv3(self.cv2(self.cv1(x)))
        return main

    def fuse(self):
        self.cv1.fuse()
        self.cv2.fuse()