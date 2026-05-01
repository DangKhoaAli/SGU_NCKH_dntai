import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.cross_view_attention import CrossViewAttention
from modules.tag_classifier import TagClassifier


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)

        # --- Cross-View Attention Fusion (bật qua --use_cross_view_attention) ---
        self.use_cross_view_attn = getattr(args, 'use_cross_view_attention', False)
        if self.use_cross_view_attn:
            d_vf = getattr(args, 'd_vf', 2048)
            num_heads = getattr(args, 'cross_view_num_heads', 8)
            self.cross_view_attn = CrossViewAttention(
                d_vf=d_vf,
                num_heads=num_heads,
                dropout=getattr(args, 'dropout', 0.1),
            )

        # --- Auxiliary Tag Classifier (bật qua --use_tag_loss) ---
        self.use_tag_loss = getattr(args, 'use_tag_loss', False)
        if self.use_tag_loss:
            # fc_feats = cat(fc_0, fc_1) → dim = 2 * d_vf = 4096
            fc_dim = 2 * getattr(args, 'd_vf', 2048)
            num_tags = getattr(args, 'num_tags', 14)
            self.tag_classifier = TagClassifier(
                fc_dim=fc_dim,
                num_tags=num_tags,
                hidden=getattr(args, 'tag_hidden_dim', 1024),
                dropout=getattr(args, 'tag_dropout', 0.3),
            )

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        # Cross-View Attention: cho 2 view "hỏi" nhau trước khi concat
        if self.use_cross_view_attn:
            att_feats_0, att_feats_1 = self.cross_view_attn(att_feats_0, att_feats_1)

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)   # [B, 4096]
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1) # [B, 98, 2048]

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            if self.use_tag_loss:
                tag_logits = self.tag_classifier(fc_feats)
                return output, tag_logits
            return output, None
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, images, targets=None, mode='train', update_opts={}):
        if self.args.dataset_name == 'iu_xray':
            return self.forward_iu_xray(images, targets, mode, update_opts)
        return self.forward_mimic_cxr(images, targets, mode, update_opts)

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            if self.use_tag_loss:
                # fc_feats cho mimic là [B, 2048] — cần expand để phù hợp TagClassifier
                # Đơn giản nhất: pad 0 hoặc dùng fc_dim riêng; tạm thời tắt tag loss cho mimic
                return output, None
            return output, None
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError(f"Unknown mode: {mode}")
