import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


# inspired and borrow from MVT, thanks Shuo Chen
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim=None, num_heads=None, qkv_bias=None, attn_drop=None, proj_drop=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (self.dim // self.num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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

        return  x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads = heads, qkv_bias=None, attn_drop=0., proj_drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MVT_imp(nn.Module):
    def __init__(self, img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=384, depth=12,
                 cut = 4, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.num_patches = int((img_size[0]/patch_size[0]) * (img_size[1]/patch_size[1]))
        self.patch_to_embedding1 = nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_to_embedding2 = nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_to_embedding3 = nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks11 = Transformer(dim=embed_dim, depth=int(depth-cut), heads=num_heads, mlp_dim=int(mlp_ratio*embed_dim))
        self.blocks12 = Transformer(dim=embed_dim, depth=int(depth-cut), heads=num_heads, mlp_dim=int(mlp_ratio*embed_dim))
        self.blocks13 = Transformer(dim=embed_dim, depth=int(depth-cut), heads=num_heads, mlp_dim=int(mlp_ratio*embed_dim))
        self.blocks2 = Transformer(dim=embed_dim, depth=cut, heads=num_heads, mlp_dim=int(mlp_ratio*embed_dim))

        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.f2c = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.dp = nn.Dropout(0.5)


    def forward_features(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x1 = self.patch_to_embedding1(x1).flatten(2).transpose(1, 2)
        x2 = self.patch_to_embedding2(x2).flatten(2).transpose(1, 2)
        x3 = self.patch_to_embedding3(x3).flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x1 = torch.cat((cls_token, x1), dim=1) + self.pos_embed1
        x2 = torch.cat((cls_token, x2), dim=1) + self.pos_embed2
        x3 = torch.cat((cls_token, x3), dim=1) + self.pos_embed3

        x1 = self.blocks11(x1)
        x2 = self.blocks12(x2)
        x3 = self.blocks13(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.blocks2(x)
        x = self.norm(x)
        # return x[:, 0]
        return torch.mean(x, dim=1)

    def forward(self, x):
        f = self.forward_features(x)
        c = self.dp(f)
        c = self.f2c(c)
        return c, f

###############################################################for debug this model#############################################
# x = torch.rand((1, 1, 240, 320))
# x = [x, x, x]
# model = MVT_imp()
# c, f = model(x)
# print(c.shape)
# print(f.shape)

