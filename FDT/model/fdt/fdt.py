"""
===============================================================================
Author(Copyright (C) 2025): Junduan Huang, Sushil Bhattacharjee, Sébastien Marcel, Wenxiong Kang.
Institution: 
School of Artificial Intelligence at South China Normal University, Foshan, 528225, China;
School of Automation Science and Engineering at South China University of Technology, Guangzhou, 510641, China;
Biometrics Security and Privacy Group at Idiap Research Institute, Martigny, 1920, Switzerland.

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at runrunjun@163.com
===============================================================================
"""

import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# please note that some of the code are borrow from EViT, LocalViT, BatchFormer, CPG. Thank these authors very much!
def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

# borrow from EViT, thanks
class AttentionWithDynamicOutput(nn.Module):
    def __init__(self, dim=None, num_heads=None, qkv_bias=None, attn_drop=None, proj_drop=None, keep_rate=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.keep_rate < 1:
            left_tokens = math.ceil(self.keep_rate * (N - 1))
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens
        else:
            left_tokens = N - 1
            return x, None, None, None, left_tokens

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, stride=None, expand_ratio=None, act=None, reduction=None, wo_dp_conv=None, dp_first=None):
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.lffn = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.lffn(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features=None, hidden_features=None, out_features=None, act_layer=None, drop=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# the most key module in this model, with two key part: AttentionWithDynamicOutput and LocalityFeedForward
class DynamicTransformerWithLocalEnhanceBlockNPEG(nn.Module):
    def __init__(self, dim=None, num_heads=None, mlp_ratio=None, qkv_bias=None, drop=None, attn_drop=None,
                 act_layer=None, norm_layer=None, keep_rate=None, fuse_token=None, use_lffn=None, use_peg=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.keep_rate = keep_rate
        self.fuse_token = fuse_token
        self.awdo = AttentionWithDynamicOutput(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, keep_rate=self.keep_rate)
        self.lffn = LocalityFeedForward(in_dim=dim, out_dim=dim, stride=1, expand_ratio=mlp_ratio, act='hs+se', reduction=dim//4, wo_dp_conv=False, dp_first=False)

        # for ablation of lffn
        self.use_lffn = use_lffn
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.peg = PEG_everyblock()
        self.use_peg = use_peg

    def forward(self, x):
        B, N, C = x.shape
        if self.fuse_token and (N!=101)  and (N!=151) and (N!=301):
            x_patch = x[:, 1:-1]
            x_others = x[:, -1]
            x_others = torch.unsqueeze(x_others, 1)
        else:
            x_patch = x[:, 1:]
            x_others = None
        x_cls = x[:, 0:1]
        
        if self.use_peg:
            pe = self.peg(x_patch)
            x_patch = x_patch + pe
            
        if x_others is not None:
            x = torch.cat([x_cls, x_patch, x_others], dim=1)
        else:
            x = torch.cat([x_cls, x_patch], dim=1)
        tmp, index, idx, cls_attn, left_tokens = self.awdo(self.norm1(x), self.keep_rate)
        x = x + tmp

        # for extracting the keep tokens and fusing the others tokens to a extra token
        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                # x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)  # becaus want to use the lffn, so the extra_token is better be add later
                x = torch.cat([x[:, 0:1], x_others], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        if self.use_lffn:
            B, N, C = x.shape # this is the new x (only the topk tokens and the cls token)
            square_size = int(math.sqrt(N-1))
            cls_token = x[:, 0].unsqueeze(1)
            x = x[:, 1:]
            if N == 301:
                x = rearrange(x, 'b (h w) c -> b c h w', h=20, w=15)
            else:
                x = x.transpose(1, 2).view(B, C, square_size, square_size)
            x = self.lffn(x).flatten(2).transpose(1, 2)
            x = torch.cat([cls_token, x], dim=1)
        else:
            x = x + self.mlp(self.norm2(x))

        if self.fuse_token and self.keep_rate<1:
            x = torch.cat([x, extra_token], dim=1)
        else:
            pass

        n_tokens = x.shape[1] - 1
        return x, n_tokens

# with two key part: AttentionWithDynamicOutput and LocalityFeedForward
class DynamicTransformerWithLocalEnhanceBlock(nn.Module):
    def __init__(self, dim=None, num_heads=None, mlp_ratio=None, qkv_bias=None, drop=None, attn_drop=None,
                 act_layer=None, norm_layer=None, keep_rate=None, fuse_token=None, use_lffn=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.keep_rate = keep_rate
        self.fuse_token = fuse_token
        self.awdo = AttentionWithDynamicOutput(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, keep_rate=self.keep_rate)
        self.lffn = LocalityFeedForward(in_dim=dim, out_dim=dim, stride=1, expand_ratio=mlp_ratio, act='hs+se', reduction=dim//4, wo_dp_conv=False, dp_first=False)

        # for ablation of lffn
        self.use_lffn = use_lffn
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        B, N, C = x.shape
        tmp, index, idx, cls_attn, left_tokens = self.awdo(self.norm1(x), self.keep_rate)
        x = x + tmp

        # for extracting the keep tokens and fusing the others tokens to a extra token
        if index is not None:
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                # x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)  # becaus want to use the lffn, so the extra_token is better be add later
                x = torch.cat([x[:, 0:1], x_others], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        if self.use_lffn:
            B, N, C = x.shape # this is the new x (only the topk tokens and the cls token)
            square_size = int(math.sqrt(N-1))
            cls_token = x[:, 0].unsqueeze(1)
            x = x[:, 1:]
            x = x.transpose(1, 2).view(B, C, square_size, square_size)
            x = self.lffn(x).flatten(2).transpose(1, 2)
            x = torch.cat([cls_token, x], dim=1)
        else:
            x = x + self.mlp(self.norm2(x))

        if self.fuse_token and self.keep_rate<1:
            x = torch.cat([x, extra_token], dim=1)
        else:
            pass

        n_tokens = x.shape[1] - 1
        return x, n_tokens

class Batch_MLP(nn.Module):
    def __init__(self, in_features=None, hidden_features=None, out_features=None, act_layer=None, drop=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Batch_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.unsqueeze(1)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.squeeze(1)
        return x



class BatchFormer(nn.Module):
    def __init__(self, dim=None, num_heads=None, mlp_ratio=None, qkv_bias=True, proj_drop=None, attn_drop=None, act_layer=None, norm_layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Batch_Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.hidden_dim = int(mlp_ratio * dim)
        self.mlp = Batch_MLP(in_features=dim, hidden_features=self.hidden_dim, out_features=dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        pre_x = x
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.mlp(x))
        x = torch.cat([pre_x, x], dim=0)
        return x

class BA(nn.Module):
    def __init__(self, dim=None, num_heads=None, mlp_ratio=None, qkv_bias=True, proj_drop=None, attn_drop=None, act_layer=None, norm_layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Batch_Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.hidden_dim = int(mlp_ratio * dim)
        self.mlp = Batch_MLP(in_features=dim, hidden_features=self.hidden_dim, out_features=dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        pre_x = x
        x = x + self.norm1(self.attn(x))
        # x = x + self.norm2(self.mlp(x))
        x = torch.cat([pre_x, x], dim=0)
        return x


class Batch_Attention_cos(nn.Module):
    def __init__(self, attn_drop=None, proj_drop=None):
        super().__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.dp = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C = x.shape
        q, k, v = x, x, x

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).t()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).reshape(B, C)
        x = self.dp(x)

        return x

class BAcos(nn.Module):
    def __init__(self, dim=None, attn_drop=None, proj_drop=None, norm_layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Batch_Attention_cos(attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        pre_x = x
        x_attn = self.attn(self.norm1(x))  # no need to do the add operation and the ffn operation
        x = torch.cat([pre_x, x_attn], dim=0)

        return x


class PEG_everyblock(nn.Module):
    def __init__(self):
        super(PEG_everyblock, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True), )

    def forward(self, x):
        B, N, C = x.shape
        if N == 150:
            pe = rearrange(x, 'b (h w) c -> b c h w', h=10, w=15)
        elif N == 300:
            pe = rearrange(x, 'b (h w) c -> b c h w', h=20, w=15)
        else:
            pe = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(N)), w=int(math.sqrt(N)))
        pe = self.proj(pe)
        pe = rearrange(pe, 'b c h w -> b (h w) c')
        return pe
        
class Patch_embedding(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, hid_channels=None, patch_size=None):
        super(Patch_embedding, self).__init__()
        self.layer_1 = nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=patch_size, stride=patch_size)
        self.layer_2 = nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        
        return x

class FDT(nn.Module):
    def __init__(self, img_size=None, patch_size=None, in_chans=None, num_classes=None, embed_dim=None, depth_fv=None, depth_sv=None,
                 num_heads=None, mlp_ratio=None, qkv_bias=True, drop_rate=None, attn_drop_rate=None, norm_layer=None,
                 act_layer=None, keep_rate_fv=None, keep_rate_sv=None, fuse_token=False, use_lffn=True, use_ba=True, batch_train=None, use_fa=None, use_mlppatch=None, use_peg=None):
        """

        """
        super().__init__()
        # about the patch embedding
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_chans
        self.embed_dim = embed_dim
        self.num_patches = int((self.img_size[0]/self.patch_size[0]) * (self.img_size[1]/self.patch_size[1])*3)
        if use_mlppatch:
            self.patch_embed_v1 = Patch_embedding(in_channels=self.in_channels, out_channels=self.embed_dim, hid_channels=384, patch_size=self.patch_size)
            self.patch_embed_v2 = Patch_embedding(in_channels=self.in_channels, out_channels=self.embed_dim, hid_channels=384, patch_size=self.patch_size)
            self.patch_embed_v3 = Patch_embedding(in_channels=self.in_channels, out_channels=self.embed_dim, hid_channels=384, patch_size=self.patch_size)
        else:
            self.patch_embed_v1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            self.patch_embed_v2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            self.patch_embed_v3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        


        # about the dynamic transformer
        self.keep_rate_fv = keep_rate_fv
        self.keep_rate_sv = keep_rate_sv
        self.depth_fv = depth_fv
        self.depth_sv = depth_sv
        self.fuse_token = fuse_token

        # about the general transformer params
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.num_features = self.embed_dim
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = self.norm_layer(self.embed_dim)
        self.act_layer = act_layer or nn.GELU
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # about the key block
        self.use_lffn = use_lffn
        self.use_peg = use_peg
        self.blocks = nn.ModuleList([
            DynamicTransformerWithLocalEnhanceBlockNPEG(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, norm_layer=self.norm_layer, act_layer=self.act_layer,
                keep_rate=self.keep_rate_fv[i], fuse_token=self.fuse_token, use_lffn=self.use_lffn, use_peg=self.use_peg)
            for i in range(self.depth_fv)])
        self.blocks_v1 = nn.ModuleList([
            DynamicTransformerWithLocalEnhanceBlockNPEG(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, norm_layer=self.norm_layer, act_layer=self.act_layer,
                keep_rate=self.keep_rate_sv[i], fuse_token=self.fuse_token, use_lffn=self.use_lffn, use_peg=self.use_peg)
            for i in range(self.depth_sv)])
        self.blocks_v2 = nn.ModuleList([
            DynamicTransformerWithLocalEnhanceBlockNPEG(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, norm_layer=self.norm_layer, act_layer=self.act_layer,
                keep_rate=self.keep_rate_sv[i], fuse_token=self.fuse_token, use_lffn=self.use_lffn, use_peg=self.use_peg)
            for i in range(self.depth_sv)])
        self.blocks_v3 = nn.ModuleList([
            DynamicTransformerWithLocalEnhanceBlockNPEG(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, norm_layer=self.norm_layer, act_layer=self.act_layer,
                keep_rate=self.keep_rate_sv[i], fuse_token=self.fuse_token, use_lffn=self.use_lffn, use_peg=self.use_peg)
            for i in range(self.depth_sv)])
        # if batch_train == BatchFormer:
        #     self.ba = batch_train(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, proj_drop=self.drop_rate, attn_drop=self.attn_drop_rate, act_layer=self.act_layer, norm_layer=self.norm_layer)
        if batch_train == BA:
            self.ba = batch_train(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, proj_drop=self.drop_rate, attn_drop=self.attn_drop_rate, act_layer=self.act_layer, norm_layer=self.norm_layer)
        if batch_train == BAcos:
            self.ba = batch_train(dim=self.embed_dim, attn_drop=self.attn_drop_rate, proj_drop=self.drop_rate, norm_layer=self.norm_layer)
        self.use_ba= use_ba

        # about the classifier
        self.num_classes = num_classes
        self.f2c = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.dp = nn.Dropout(0.5)

        # about the feature attention base on patches
        self.use_fa = use_fa
        self.fc1 = nn.Linear(128, 64)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 128)
        self.act2 = nn.Sigmoid()


    def patch_embedding_single_view_dt(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        x1 = self.patch_embed_v1(x1).flatten(2).transpose(1, 2)
        x2 = self.patch_embed_v2(x2).flatten(2).transpose(1, 2)
        x3 = self.patch_embed_v3(x3).flatten(2).transpose(1, 2)

        _, _, x1 = self.dt_v1(x1)
        _, _, x2 = self.dt_v2(x2)
        _, _, x3 = self.dt_v3(x3)
        x = torch.cat([x1, x2, x3], dim=1)

        return x
    
    def dt_v1(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        left_tokens = []
        for i, blk in enumerate(self.blocks_v1):
            x, left_token = blk(x)
            left_tokens.append(left_token)
        x = self.norm(x)
        return x[:, 0], left_token, x[:, 1:]

    def dt_v2(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        left_tokens = []
        for i, blk in enumerate(self.blocks_v2):
            x, left_token = blk(x)
            left_tokens.append(left_token)
        x = self.norm(x)
        return x[:, 0], left_token, x[:, 1:]

    def dt_v3(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        left_tokens = []
        for i, blk in enumerate(self.blocks_v3):
            x, left_token = blk(x)
            left_tokens.append(left_token)
        x = self.norm(x)
        return x[:, 0], left_token, x[:, 1:]

    def dt_fv(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        left_tokens = []
        for i, blk in enumerate(self.blocks):
            x, left_token = blk(x)
            left_tokens.append(left_token)
        x = self.norm(x)
        return x[:, 0], left_token, x[:, 1:]

    def feature_attention(self, patches):
        a = patches.mean(1)
        a = self.act1(self.fc1(a))
        a = self.act2(self.fc2(a))
        return a

    def forward(self, x):
        x = self.patch_embedding_single_view_dt(x)

        f, left_token, patches = self.dt_fv(x)

        if self.use_fa:
            a = self.feature_attention(patches)
            f = f * a
        else:
            pass

        if self.use_ba & self.training:
            f = self.ba(f)
        c = self.dp(f)
        c = self.f2c(c)
        return c, f

#######################################for debug this model####################################################
# x = torch.rand((1, 1, 240, 320))
# x = [x, x, x]
# model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(300/300, 300/300, 300/300, 300/300, 300/300, 300/300), keep_rate_sv=(100/100, 100/100, 100/100, 100/100), fuse_token=True, use_fa=False,
#             use_ba=True, batch_train=BA,
#             use_lffn=True,
#             use_mlppatch=True,
#             use_peg=True)
# c, f = model(x)
# print(c.shape)
# print(f.shape)
