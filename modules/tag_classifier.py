"""
tag_classifier.py
MLP để phân loại nhãn bệnh lý (multi-label) từ global visual features.

Input : fc_feats [B, fc_dim]  (global features sau khi cat 2 view)
Output: [B, num_tags]          (logits, chưa qua sigmoid)
"""

import torch.nn as nn


class TagClassifier(nn.Module):
    """
    2-layer MLP cho multi-label tag prediction.

    Args:
        fc_dim   : chiều của global feature đầu vào (mặc định 4096 = 2×2048)
        num_tags : số nhãn bệnh lý (mặc định 14)
        hidden   : chiều lớp ẩn (mặc định 1024)
        dropout  : tỷ lệ dropout (mặc định 0.3)
    """

    def __init__(
        self,
        fc_dim: int = 4096,
        num_tags: int = 14,
        hidden: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fc_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_tags),
            # Không có Sigmoid ở đây —
            # BCEWithLogitsLoss sẽ tự xử lý để ổn định số học hơn
        )

    def forward(self, fc_feats):
        """
        Args:
            fc_feats: [B, fc_dim]
        Returns:
            logits: [B, num_tags]
        """
        return self.mlp(fc_feats)
