"""
cross_view_attention.py
Cross-View Attention Fusion cho IU-Xray (2 views: Frontal + Lateral).

Thay vì torch.cat đơn thuần, module này cho phép mỗi view
"hỏi" thông tin từ view kia qua Multi-Head Cross-Attention,
giúp encoder hiểu mối quan hệ không gian giữa 2 góc nhìn.
"""

import torch
import torch.nn as nn


class CrossViewAttention(nn.Module):
    """
    Bidirectional Cross-Attention giữa 2 feature maps.

    Input:
        patch_0: [B, N0, C]  (patch features từ view 0 - Frontal)
        patch_1: [B, N1, C]  (patch features từ view 1 - Lateral)

    Output:
        enhanced_0: [B, N0, C]  (patch_0 sau khi attend sang patch_1)
        enhanced_1: [B, N1, C]  (patch_1 sau khi attend sang patch_0)
    """

    def __init__(self, d_vf: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 0→1: frontal queries lateral
        self.attn_0to1 = nn.MultiheadAttention(
            embed_dim=d_vf,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 1→0: lateral queries frontal
        self.attn_1to0 = nn.MultiheadAttention(
            embed_dim=d_vf,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_0 = nn.LayerNorm(d_vf)
        self.norm_1 = nn.LayerNorm(d_vf)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patch_0: torch.Tensor,
        patch_1: torch.Tensor,
    ):
        """
        Args:
            patch_0: [B, N, C]
            patch_1: [B, N, C]

        Returns:
            enhanced_0: [B, N, C]
            enhanced_1: [B, N, C]
        """
        # Frontal attend to Lateral
        ctx_0, _ = self.attn_0to1(
            query=patch_0,
            key=patch_1,
            value=patch_1,
        )
        enhanced_0 = self.norm_0(patch_0 + self.dropout(ctx_0))

        # Lateral attend to Frontal
        ctx_1, _ = self.attn_1to0(
            query=patch_1,
            key=patch_0,
            value=patch_0,
        )
        enhanced_1 = self.norm_1(patch_1 + self.dropout(ctx_1))

        return enhanced_0, enhanced_1
