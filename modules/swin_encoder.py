# modules/swin_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class CrossViewAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        if N != 98: return x
        v1 = x[:, :49, :] 
        v2 = x[:, 49:, :] 
        attn_v1, _ = self.attn(v1, v2, v2)
        v1 = v1 + self.dropout(attn_v1)
        attn_v2, _ = self.attn(v2, v1, v1)
        v2 = v2 + self.dropout(attn_v2)
        x = torch.cat([v1, v2], dim=1)
        return self.norm(x)

class SwinBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size=3, shift=False):
        super(SwinBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        if N == 98: H, W = 7, 14
        else: H = W = int(N**0.5)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # --- PADDING ĐỂ CHIA HẾT CHO WINDOW_SIZE ---
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # 1. Shifted Window
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # 2. Window Partition
        x_windows = window_partition(x, self.window_size) 

        # 3. Attention (Không dùng mask padding ở đây vì ảnh y tế thường đầy đủ)
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)

        # 4. Window Merge
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 5. Reverse Shift
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        # 6. UNPAD (Cắt bỏ phần đệm để về lại 7x7 hoặc 7x14)
        x = x[:, :H, :W, :].contiguous()

        x = x.view(B, N, C)
        x = x + shortcut
        x = x + self.ffn(self.norm2(x))

        return x

class SwinEncoder(nn.Module):
    def __init__(self, d_model=512, num_layers=3, num_heads=8, window_size=3):
        super().__init__()
        self.cross_view = CrossViewAttentionBlock(d_model, num_heads)
        self.layers = nn.ModuleList([
            SwinBlock(d_model, num_heads, window_size=window_size, shift=(i % 2 == 1))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.cross_view(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
