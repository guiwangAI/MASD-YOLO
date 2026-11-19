import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SHSA', 'SHSA_V2', 'C2f_SHSA1', 'C2f_SHSA2']


class SHSA(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(SHSA, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction

        self.conv_q = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        
        self.conv_k = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)       

        self.conv_v = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        
        self.conv_attn = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)


        self.tanh = nn.Tanh()

    def forward(self, x, A=None, alpha=1):

        B, C, H, W = x.shape
        
        q = self.conv_q(x).mean(-2)
        k = self.conv_k(x).mean(-2)

        attn = self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))
        
        attn = self.conv_attn(attn) * alpha
        
        if A is not None:
            attn = attn + A.unsqueeze(0).unsqueeze(0)

        v = self.conv_v(x)

        output = torch.einsum('ncuv,nctv->nctu', attn, v)

        return output


class SHSA_V2(nn.Module):

    def __init__(self, in_channels, out_channels, rel_reduction=8, use_softmax=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_softmax = use_softmax
        
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        

        self.conv_q_h = nn.Conv2d(in_channels, self.rel_channels, 1)
        self.conv_k_h = nn.Conv2d(in_channels, self.rel_channels, 1)
        
        self.conv_q_w = nn.Conv2d(in_channels, self.rel_channels, 1)
        self.conv_k_w = nn.Conv2d(in_channels, self.rel_channels, 1)
        
        self.conv_v = nn.Conv2d(in_channels, out_channels, 1)
        
        self.conv_attn_h = nn.Conv2d(self.rel_channels, out_channels, 1)
        self.conv_attn_w = nn.Conv2d(self.rel_channels, out_channels, 1)
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.act = nn.Tanh() if not use_softmax else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        q_h = self.conv_q_h(x).mean(-2)
        k_h = self.conv_k_h(x).mean(-2)
        
        attn_h = q_h.unsqueeze(-1) - k_h.unsqueeze(-2)
        attn_h = self.act(attn_h) if not self.use_softmax else attn_h
        
        if self.use_softmax:
            attn_h = F.softmax(attn_h / self.temperature, dim=-1)
        
        attn_h = self.conv_attn_h(attn_h)     

        q_w = self.conv_q_w(x).mean(-1)
        k_w = self.conv_k_w(x).mean(-1)
        
        attn_w = q_w.unsqueeze(-1) - k_w.unsqueeze(-2)
        attn_w = self.act(attn_w) if not self.use_softmax else attn_w
        
        if self.use_softmax:
            attn_w = F.softmax(attn_w / self.temperature, dim=-1)
        
        attn_w = self.conv_attn_w(attn_w)
        
        v = self.conv_v(x)
        
        out = torch.einsum('ncuv,nctv->nctu', attn_h, v)
        
        out = torch.einsum('ncuv,ncuT->ncvT', attn_w, out)
        
        return out



def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution"""
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck_SHSA1(nn.Module):
    """Bottleneck with SHSA - Version 1"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.att = SHSA_V2(c_, c_, rel_reduction=8, use_softmax=False)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.att(self.cv1(x)))


class Bottleneck_SHSA2(nn.Module):
    """Bottleneck with SHSA - Version 2"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.att = SHSA_V2(c_, c_, rel_reduction=8, use_softmax=False)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.att(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class C2f_SHSA1(nn.Module):
    """C2f with SHSA - Version 1"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_SHSA1(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_SHSA2(nn.Module):
    """C2f with SHSA - Version 2"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_SHSA2(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))