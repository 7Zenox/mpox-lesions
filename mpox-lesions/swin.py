import torch
import torch.nn as nn
from einops import rearrange


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super(SwinBlock, self).__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        B, H, W, C = x.shape
        x = rearrange(x, "b h w c -> b (h w) c")  # Flatten spatial dimensions
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)  # Restore spatial dimensions
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=6,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
    ):
        super(SwinTransformer, self).__init__()
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=4, stride=4)
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.Sequential(
                *[SwinBlock(embed_dim * (2**i), num_heads[i]) for _ in range(depths[i])]
            )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(embed_dim * (2 ** (len(depths) - 1)))
        self.head = nn.Linear(embed_dim * (2 ** (len(depths) - 1)), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # Patch embedding
        x = rearrange(x, "b c h w -> b h w c")  # Rearrange for Swin blocks
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, "b h w c -> b (h w) c")  # Flatten spatial dimensions
        x = self.norm(x.mean(dim=1))  # Global average pooling
        x = self.head(x)
        return x
