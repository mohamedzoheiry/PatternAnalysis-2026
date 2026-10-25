import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """Small 4-block CNN baseline for AD vs NC (~0.39M parameters)."""

    def __init__(self, in_ch=1, num_classes=2, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (in_ch, *widths)
        self.features = nn.Sequential(
            *[ConvBlock(a, b) for a, b in zip(chans, chans[1:])]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], num_classes)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


# GFNet


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.grid_size = img_size // patch_size  # tokens per side
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, D, 14, 14)
        return x.flatten(2).transpose(1, 2)  # (B, 196, D)


class GlobalFilter(nn.Module):
    def __init__(self, dim, grid_size=14):
        super().__init__()
        self.h = grid_size
        self.w = grid_size // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2) * 0.02)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.h, self.h, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(self.h, self.h), dim=(1, 2), norm="ortho")
        return x.reshape(B, N, C)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    """GFNet block: LayerNorm -> GlobalFilter -> LayerNorm -> MLP"""

    def __init__(self, dim, grid_size, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.filter = GlobalFilter(dim, grid_size)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))


class GFNet(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=256,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = self.patch_embed.grid_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, grid, mlp_ratio, drop_rate, dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Pooled feature vector before the head, (B, D)."""
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        x = self.blocks(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
