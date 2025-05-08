import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from typing import Any


def conv1x1(in_channels, out_channels, stride=1, padding=0, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                     stride=stride, padding=padding, *args, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, padding=1, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, *args, **kwargs)

class mapFunction(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(mapFunction, self).__init__()
        inp = int(inp)
        oup = int(oup)
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw 60w
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # dw 30w
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=16, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # pw-linear 60w
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # conv1x1(hidden_dim, oup),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class IAB(nn.Module):
    def __init__(self, in_channels, out_channels, iter=1):
        super(IAB, self).__init__()
        self.n_blocks = iter
        self.alpha = nn.ModuleList()
        self.s = nn.ModuleList()
        self.t = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.alpha.append(mapFunction(inp=in_channels, oup=out_channels, expand_ratio=1))
            self.s.append(mapFunction(inp=in_channels, oup=out_channels, expand_ratio=1))
            self.t.append(mapFunction(inp=in_channels, oup=out_channels, expand_ratio=1))

    def forward(self, z1, z2):
        # 进行格式重排(Transformer将数据变换为:[batch, w, h, head]),图像数据为[batch, head/c, w, h]
        z1 = z1.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)

        for i in range(self.n_blocks):
            z2 = z2 + self.alpha[i](z1)
            z1 = z1 * torch.exp(self.s[i](z2)) + self.t[i](z2)
            # z1, z2 = z2, z1

        z1 = z1.permute(0, 2, 3, 1)
        z2 = z2.permute(0, 2, 3, 1)

        return z1, z2

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, cross_attn, h_attn, iab_iter):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.cross_attn = cross_attn
        if self.cross_attn:
            self.iab = IAB(inner_dim, inner_dim, iab_iter)
        self.h_attn = h_attn

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(self.create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(self.create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        if not self.cross_attn:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = self.get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def create_mask(self, window_size, displacement, upper_lower, left_right):
        mask = torch.zeros(window_size ** 2, window_size ** 2)

        if upper_lower:
            mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
            mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

        if left_right:
            mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
            mask[:, -displacement:, :, :-displacement] = float('-inf')
            mask[:, :-displacement, :, -displacement:] = float('-inf')
            mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

        return mask

    def get_relative_distances(self, window_size):
        indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        distances = indices[None, :, :] - indices[:, None, :]
        return distances

    def forward(self, x, y=None):

        if self.shifted:
            x = self.cyclic_shift(x)
            if self.cross_attn:
                y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # print('forward-x: ', x.shape)   # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
        if not self.cross_attn:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # [N, H//downscaling_factor, W//downscaling_factor, head_dim * heads] * 3
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            k = kv[0]
            v = kv[1]
            q = self.to_q(y)

            # IAB
            k_iab, q_iab = self.iab(k, q)
            # 高频映射
            if self.h_attn:
                k = k - k_iab
                q = q_iab
            else:
                k = k_iab
                q = q_iab

            qkv = (q, k, v)

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        # print('forward-q: ', q.shape)   # [N, num_heads, num_win, win_area, hidden_dim/num_heads]
        # print('forward-k: ', k.shape)
        # print('forward-v: ', v.shape)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q * k / sqrt(d)

        # 更改.relative_indices类型
        self.relative_indices = self.relative_indices.long()
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head]
        out = self.to_out(out)
        # [N, H//downscaling_factor, W//downscaling_factor, dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn, h_attn, iab_iter):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn,
                                                                     h_attn=h_attn,
                                                                     iab_iter=iab_iter)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]

class SwinModule(nn.Module):
    def __init__(self, in_channels=4, hidden_dimension=32, layers=2, downscaling_factor=1, num_heads=4, head_dim=16, window_size=4,
                 relative_pos_embedding=True, cross_attn=False, h_attn=False, iab_iter=1):
        r"""
        Args:
            in_channels(int): 输入通道数
            hidden_dimension(int): 隐藏层维数,patch_partition提取patch时有个Linear学习的维数
            layers(int): swin block数,必须为2的倍数,连续的regular block和shift block
            downscaling_factor: H,W上下采样倍数
            num_heads: multi-attn 的 attn 头的个数
            head_dim:   每个attn 头的维数
            window_size:    窗口大小,窗口内进行attn运算
            relative_pos_embedding: 相对位置编码
            cross_attn: 交叉注意力机制(有两个输入时一定要使用)
            h_attn: 第二个输入是否为高频信息.若为高频,则对iab的结果需要进行反转
            iab_iter: iab循环层数
        """
        super().__init__()
        # assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        layers = layers * 2
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn, h_attn=h_attn, iab_iter=iab_iter),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn, h_attn=h_attn, iab_iter=iab_iter),
            ]))

    def forward(self, x, y=None):
        if y is None:
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
        else:
            x = self.patch_partition(x)
            y = self.patch_partition(y)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x, y)
                x = shifted_block(x, y)
            return x.permute(0, 3, 1, 2)
