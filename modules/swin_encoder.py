import torch
import torch.nn as nn


def infer_feature_grid(num_patches):
    """Suy ra kich thuoc grid tu so patch dau vao."""
    if num_patches == 98:
        # IU X-ray: 2 anh, moi anh co feature map 7x7.
        # Code hien tai dat 2 feature map canh nhau thanh grid 7x14.
        return 7, 14

    grid_size = int(num_patches ** 0.5)
    return grid_size, grid_size


def check_window_compatible(height, width, num_patches, window_size):
    """Kiem tra grid co dung shape va chia het cho window_size khong."""
    if height * width != num_patches:
        raise RuntimeError(
            f"So luong patches {num_patches} khong khop voi grid "
            f"{height}x{width}. Hay kiem tra lai visual_extractor."
        )

    if height % window_size != 0 or width % window_size != 0:
        raise RuntimeError(
            f"Grid {height}x{width} khong chia het cho window_size={window_size}. "
            "Can padding feature map hoac chon window_size phu hop."
        )


def window_partition(x, window_size):
    """
    Chia feature map thanh cac window nho.

    Input:
        x: [B, H, W, C]

    Output:
        windows: [B * num_windows, window_size * window_size, C]
    """
    batch_size, height, width, channels = x.shape
    num_windows_h = height // window_size
    num_windows_w = width // window_size

    x = x.view(
        batch_size,
        num_windows_h,
        window_size,
        num_windows_w,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, channels)


def window_reverse(windows, window_size, height, width):
    """
    Ghep cac window ve lai feature map ban dau.

    Input:
        windows: [B * num_windows, window_size * window_size, C]

    Output:
        x: [B, H, W, C]
    """
    num_windows_h = height // window_size
    num_windows_w = width // window_size
    num_windows = num_windows_h * num_windows_w
    batch_size = windows.shape[0] // num_windows

    x = windows.view(
        batch_size,
        num_windows_h,
        num_windows_w,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, -1)


class SwinBlock(nn.Module):
    """
    Mot block Swin don gian:
        LayerNorm -> Window Attention -> residual
        LayerNorm -> FFN              -> residual

    Block le co the dung shifted window de cac patch gan bien window
    trao doi thong tin voi window lan can.
    """

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
            nn.Linear(d_model * 4, d_model),
        )

    def _get_shift_size(self, height, width):
        """Tinh do dich window. Chieu nao chi co 1 window thi khong can shift."""
        shift_h = 0 if height <= self.window_size else self.window_size // 2
        shift_w = 0 if width <= self.window_size else self.window_size // 2
        return shift_h, shift_w

    def _build_shift_attention_mask(self, batch_size, height, width, shift_h, shift_w, device):
        """
        Tao attention mask cho shifted window.

        Sau khi roll feature map, mot window moi co the chua patch den tu
        nhieu vung khac nhau. Mask nay chan attention giua cac vung khac nhau,
        tranh viec patch bi roll vong qua bien anh noi chuyen sai ngu canh.
        """
        region_mask = torch.zeros((1, height, width, 1), device=device)

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_h),
            slice(-shift_h, None),
        ) if shift_h > 0 else (slice(0, None),)

        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_w),
            slice(-shift_w, None),
        ) if shift_w > 0 else (slice(0, None),)

        region_id = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                region_mask[:, h_slice, w_slice, :] = region_id
                region_id += 1

        mask_windows = window_partition(region_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        # True trong attn_mask cua PyTorch co nghia la vi tri bi chan.
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask != 0

        window_area = self.window_size * self.window_size
        attn_mask = attn_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        attn_mask = attn_mask.view(-1, window_area, window_area)
        return attn_mask.repeat_interleave(self.num_heads, dim=0)

    def _build_key_padding_mask(self, mask, batch_size, height, width, num_patches, shift_h, shift_w, do_shift):
        """
        Doi mask cua project sang key_padding_mask cua MultiheadAttention.

        Project dung:
            1 = patch hop le
            0 = patch bi che

        PyTorch MultiheadAttention dung:
            False = giu lai
            True  = che di
        """
        if mask is None:
            return None

        key_mask = mask.view(batch_size, num_patches) == 0

        if do_shift:
            key_mask = key_mask.view(batch_size, height, width)
            key_mask = torch.roll(key_mask, shifts=(-shift_h, -shift_w), dims=(1, 2))
            key_mask = key_mask.view(batch_size, num_patches)

        num_windows_h = height // self.window_size
        num_windows_w = width // self.window_size
        num_windows = num_windows_h * num_windows_w

        if num_windows == 1:
            return key_mask

        key_mask = key_mask.view(
            batch_size,
            num_windows_h,
            self.window_size,
            num_windows_w,
            self.window_size,
        )
        key_mask = key_mask.permute(0, 1, 3, 2, 4).contiguous()
        return key_mask.view(-1, self.window_size * self.window_size)

    def forward(self, x, mask=None):
        """
        Input:
            x:    [B, N, C]
            mask: [B, 1, N] hoac [B, 1, 1, N], optional

        Output:
            x: [B, N, C]
        """
        batch_size, num_patches, channels = x.shape
        height, width = infer_feature_grid(num_patches)
        check_window_compatible(height, width, num_patches, self.window_size)

        shortcut = x

        # Dua patch sequence ve feature map de chia window.
        x = self.norm1(x)
        x = x.view(batch_size, height, width, channels)

        shift_h, shift_w = self._get_shift_size(height, width)
        do_shift = self.shift and (shift_h > 0 or shift_w > 0)

        if do_shift:
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)

        attn_mask = None
        if do_shift:
            attn_mask = self._build_shift_attention_mask(
                batch_size=batch_size,
                height=height,
                width=width,
                shift_h=shift_h,
                shift_w=shift_w,
                device=x.device,
            )

        key_padding_mask = self._build_key_padding_mask(
            mask=mask,
            batch_size=batch_size,
            height=height,
            width=width,
            num_patches=num_patches,
            shift_h=shift_h,
            shift_w=shift_w,
            do_shift=do_shift,
        )

        attn_windows, _ = self.attn(
            x_windows,
            x_windows,
            x_windows,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        x = window_reverse(attn_windows, self.window_size, height, width)

        if do_shift:
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

        x = x.view(batch_size, num_patches, channels)
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x


class SwinEncoder(nn.Module):
    """
    Encoder gom nhieu SwinBlock.

    Cac block chan dung window attention binh thuong.
    Cac block le dung shifted window attention.
    """

    def __init__(self, d_model=512, num_layers=3, num_heads=8, window_size=7):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinBlock(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                shift=(layer_idx % 2 == 1),
            )
            for layer_idx in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
