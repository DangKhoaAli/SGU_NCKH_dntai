import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """ Chia ma trận thành các cửa sổ nhỏ (Window Partitioning) """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """ Gộp các cửa sổ lại thành ma trận ban đầu (Window Reverse) """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size=7, shift=False):
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
        # x shape: [B, N, C]
        B, N, C = x.shape
        H = W = int(N**0.5)
        shortcut = x
        
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 1. Shifted Window (Xoay vòng ma trận)
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # 2. Window Partition (Chia cửa sổ)
        x_windows = window_partition(x, self.window_size) # [B*num_windows, win_size*win_size, C]

        # 3. Xử lý Mask cho Attention
        # Transformer trong project này gửi mask dạng [B, 1, 1, N] hoặc [B, 1, N]
        # MultiheadAttention của PyTorch dùng key_padding_mask dạng [B, N]
        key_mask = None
        if mask is not None:
            # Chuyển từ [B, 1, 1, N] -> [B, N] và đảo ngược (True là che đi)
            key_mask = (mask.view(B, N) == 0) 
            
            # Nếu có chia nhiều window, cần repeat mask cho từng window
            num_windows = x_windows.shape[0] // B
            if num_windows > 1:
                # Chia mask [B, N] thành [B*num_windows, window_size*window_size]
                key_mask = key_mask.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
                key_mask = key_mask.permute(0, 1, 3, 2, 4).contiguous().view(-1, self.window_size * self.window_size)

        # 4. Window-based Attention
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows, key_padding_mask=key_mask)

        # 5. Window Merge (Gộp cửa sổ)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # 6. Reverse Shift (Xoay ngược lại)
        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        # 7. Residual and FFN
        x = x.view(B, N, C)
        x = x + shortcut
        x = x + self.ffn(self.norm2(x))

        return x

class SwinEncoder(nn.Module):
    def __init__(self, d_model=512, num_layers=3, num_heads=8, window_size=7):
        super().__init__()
        # Đối với IU X-ray ảnh 7x7, dùng window_size=7 là hợp lý nhất
        self.layers = nn.ModuleList([
            SwinBlock(d_model, num_heads, window_size=window_size, shift=(i % 2 == 1))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
