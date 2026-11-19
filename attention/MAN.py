import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MAB', 'C2f_MAN1', 'C2f_MAN2']

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first format"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class GroupGLKA(nn.Module):
    def __init__(self, n_feats, k=3):
        super().__init__()
        
        self.n_feats = n_feats
        

        if n_feats % 3 != 0:
            adjusted_feats = ((n_feats // 3) + 1) * 3
            self.channel_adjust = nn.Conv2d(n_feats, adjusted_feats, 1, bias=False)
            self.channel_restore = nn.Conv2d(adjusted_feats, n_feats, 1, bias=False)
            n_feats = adjusted_feats
        else:
            self.channel_adjust = nn.Identity()
            self.channel_restore = nn.Identity()
        
        i_feats = 2 * n_feats
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        

        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups=n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 9, stride=1, padding=(9//2)*4, groups=n_feats//3, dilation=4),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups=n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 7, stride=1, padding=(7//2)*3, groups=n_feats//3, dilation=3),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups=n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 5, stride=1, padding=(5//2)*2, groups=n_feats//3, dilation=2),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        

        self.X3 = nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups=n_feats//3)
        self.X5 = nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups=n_feats//3)
        self.X7 = nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups=n_feats//3)
        
        self.proj_first = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.proj_last = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        
    def forward(self, x):
        shortcut = x.clone()
        
        x = self.channel_adjust(x)
        x = self.norm(x)
        x = self.proj_first(x)
        
        a, x = torch.chunk(x, 2, dim=1) 
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        
        a = torch.cat([
            self.LKA3(a_1) * self.X3(a_1),
            self.LKA5(a_2) * self.X5(a_2),
            self.LKA7(a_3) * self.X7(a_3)
        ], dim=1)
        
        x = self.proj_last(x * a) * self.scale
        x = self.channel_restore(x)
        
        return x + shortcut


class SGAB(nn.Module):
    def __init__(self, n_feats):   
        super().__init__()
        i_feats = n_feats * 2
        
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0) 
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups=n_feats)     
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
    def forward(self, x):      
        shortcut = x.clone()
        

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1) 
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        
        return x * self.scale + shortcut


class MAB(nn.Module):
    def __init__(self, n_feats):   
        super().__init__()
        
        self.LKA = GroupGLKA(n_feats)       

        self.LFE = SGAB(n_feats)       
    def forward(self, x): 

        x = self.LKA(x)  
        
        x = self.LFE(x)  
        
        return x



def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BN and activation"""
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck_MAN1(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.att = MAB(c_)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.att(self.cv1(x)))


class Bottleneck_MAN2(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.att = MAB(c_)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.att(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class C2f_MAN1(nn.Module):
    """
    C2f module with complete MAN attention - Version 1
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_MAN1(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_MAN2(nn.Module):
    """
    C2f module with complete MAN attention - Version 2
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_MAN2(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))