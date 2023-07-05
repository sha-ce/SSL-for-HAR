import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import sslearning.augmentation as aug

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()*(-math.log(10000.0)/embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class PatchEmbed(nn.Module):
    """ signal window to Patch Embedding
    """
    def __init__(self, window_size=300, patch_size=30, in_chans=3, embed_dim=768, is_eva=False, task=False):
        super().__init__()
        num_patches = window_size // patch_size
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.is_eva = is_eva
        self.task = task
        
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x, y=None):
        B, C, W = x.shape
        if not self.is_eva:
            x, mask = self.data_trans(x)
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x, mask
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x, None
    
    def data_trans(self, x, y=None):
        delta = self.patch_size
        
        d = x.device.type+':'+str(x.device.index)
        if d != 'cpu:None':
            x = x.cpu().numpy()
            y = y.cpu().numpy() if y is not None else y
        patches = [x[:,:,i*delta:i*delta+delta].transpose(0,2,1) for i in range(self.num_patches)]
        
        r = random.sample(range(self.num_patches), self.task.num)
        c = 0
        
        mask = None
        if self.task.mask: # mask
            patches[r[c]] = np.zeros((patches[r[c]].shape))
            mask = torch.ones(len(patches)+1)
            mask[r[c]+1] = 0
            mask = mask.repeat(len(patches)+1, 1)
            c+=1
        if self.task.permute: # permutation
            patches[r[c]] = aug.permutation(patches[r[c]])
            c+=1
        if self.task.timewarp: # time warp
            patches[r[c]] = aug.time_warp(patches[r[c]])
            c+=1
        if self.task.rotation: # rotation
            patches[r[c]] = aug.rotation(patches[r[c]])
            c+=1
        if self.task.wdba: # wdba
            patches[r[c]] = aug.wdba(patches[r[c]], y)
            c+=1
        if self.task.rgw: # random_guided_warp
            patches[r[c]] = aug.random_guided_warp(patches[r[c]], y)
            c+=1
        if self.task.dgw: # discriminative_guided_warp
            patches[r[c]] = aug.discriminative_guided_warp(patches[r[c]], y)
            c+=1
        
        tensor_patches = [torch.from_numpy(p) for p in patches]
        patches_cat = torch.cat(tensor_patches, axis=1).permute(0,2,1).float()
        return patches_cat.to(d), mask.to(d)



class EvaClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(embed_dim, embed_dim)
        self.linear2 = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class TimeSeriesHeadLinear(nn.Module):
    def __init__(self, embed_dim=768):
        super(TimeSeriesHeadLinear, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, 3)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class TimeSeriesHead(nn.Module):
    def __init__(self, embed_dim=768, output_dim=300):
        super(TimeSeriesHead, self).__init__()
        self.linears = [TimeSeriesHeadLinear(embed_dim).to('cuda:0') for _ in range(output_dim)]
    
    def forward(self, f):
        x = [linear(f) for linear in self.linears]
        return torch.stack(x, dim=0).permute(1,2,0)


class Transformer(nn.Module):
    def __init__(
        self,
        window_size=300,
        patch_size=30,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        is_eva=False,
        task=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            window_size=window_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            is_eva=is_eva,
            task=task
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, dropout=drop_rate)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # fully-combined layer
        if is_eva:
            self.head = EvaClassifier(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = TimeSeriesHead(embed_dim, window_size)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def prepare_tokens(self, x):
        B, nc, w = x.shape
        x, mask = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.positional_encoding(x)
        
        return self.pos_drop(x), mask
    
    def forward(self, x):
        x, mask = self.prepare_tokens(x)
        
        # attention
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        
        # head
        y = self.head(x[:,0])
        return y


    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output