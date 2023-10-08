import torch
import torch.nn as nn

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import pdb
## LGCR block

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
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

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, fmap_size, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        self.pos_bias = nn.Embedding(fmap_size[0] * fmap_size[1], heads)

        q_range_height  = torch.arange(0, fmap_size[0], step = 1)
        q_range_width   = torch.arange(0, fmap_size[1], step = 1)
        k_range_height  = torch.arange(0, fmap_size[0], step = kv_proj_stride) #+ (kv_proj_stride-1)/2
        k_range_width   = torch.arange(0, fmap_size[1], step = kv_proj_stride) #+ (kv_proj_stride-1)/2
        
        q_pos = torch.stack(torch.meshgrid(q_range_height, q_range_width), dim = -1)
        k_pos = torch.stack(torch.meshgrid(k_range_height, k_range_width), dim = -1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size[1]) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
 
        dots = self.apply_pos_bias(dots)
        
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class LGCR_block(nn.Module):
    def __init__(self, dim, proj_kernel, fmap_size, kv_proj_stride=2, depth=2, heads=2, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super(LGCR_block, self).__init__()
        self.norm = LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, fmap_size=fmap_size, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        x = self.norm(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DDC(nn.Module):    
    def __init__(self, inplanes, outplanes, kernel_size=3, downscale_factor=2):
        super(DDC, self).__init__()
        planes = inplanes * downscale_factor**2
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        
        self.conv = nn.Conv2d(planes, outplanes, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

## MFM block

class MFM_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(MFM_block, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return self.bn(torch.max(out[0], out[1]))

## variational information bottleneck head

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VIB_head(nn.Module):
    def __init__(self, feature_dim, drop_ratio = 0.4):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.BatchNorm2d(192, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            MFM_block(
                in_channels=5*5*192, 
                out_channels=feature_dim,
                type=0))

        self.logvar_head = nn.Sequential(
            nn.BatchNorm2d(192, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            MFM_block(
                in_channels=5*5*192, 
                out_channels=feature_dim,
                type=0))

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return (mu + sampler, sampler)

    def forward(self, x):
        mu = self.mu_head(x)
        if self.training:
            logvar = self.logvar_head(x)
            embedding, sampler = self._reparameterize(mu, logvar)
            return mu, logvar, embedding
        else:
            return mu



class LGCR(nn.Module):
    def __init__(self, feature_dim = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            MFM_block(
                in_channels=1, 
                out_channels=48, 
                kernel_size=9, 
                stride=1, 
                padding=0, 
                type=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.cnn_branch = nn.Sequential(
            MFM_block(
                in_channels=48, 
                out_channels=96, 
                kernel_size=5, 
                stride=1, 
                padding=0, 
                type=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            MFM_block(
                in_channels=96, 
                out_channels=128, 
                kernel_size=5, 
                stride=1, 
                padding=0, 
                type=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            MFM_block(
                in_channels=128, 
                out_channels=192, 
                kernel_size=4, 
                stride=1, 
                padding=0, 
                type=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.trans_branch = nn.Sequential(
            LGCR_block(
                dim=48, 
                proj_kernel=3, 
                fmap_size=[60,60], 
                depth=1, 
                heads=1),
            DDC(48, 96, kernel_size=3, downscale_factor=2),
            LGCR_block(
                dim=96, 
                proj_kernel=3, 
                fmap_size=[28,28], 
                depth=2, 
                heads=3),
            DDC(96, 128, kernel_size=3, downscale_factor=2),
            LGCR_block(
                dim=128, 
                proj_kernel=3, 
                fmap_size=[12,12], 
                depth=2, 
                heads=6),
            DDC(128, 192, kernel_size=2, downscale_factor=2),
        )
        self.vib_head = VIB_head(feature_dim)


    def forward(self, x):
        x = self.backbone(x)
        x_cnn = self.cnn_branch(x)
        x_trans = self.trans_branch(x)
        x = x_cnn + x_trans
        return self.vib_head(x)

if __name__=='__main__':

    model = LGCR()
    input = torch.randn(2, 1, 128, 128)
    model = model.cuda()
    input = input.cuda()
    
    output = model(input)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
