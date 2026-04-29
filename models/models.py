import torch
import numpy as np
import torch.nn as nn
from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor


text_list = ['pneumothorax','pleural', 'spine', 'heart', 'hernia', 'lung', 'mediastinal', 'Cardiac', 'Bony', 'Emphysema', 'Atelectasis', 'lobe', 'clavicle', 'Cardiomediastinal', 'osseous', 'mediastinum', 'aorta', 'aortic', 'diaphragm', 'thoracic', 'vascularity', 'pulmonary' ]

text_list_mimic = ['cholecystectomy', 'subclavian', 'emphysema', 'bronchovascular', 'heart', 'neck', 'wires', 'hilum', 'mediastinum', 'cardiac', 'hemithorax', 'rib', 'sternotomy', 'chest', 'tubes', 'osseous', 'diaphragms', 'bony', 'silhouette', 'hilar', 'mediastinal', 'perihilar','vasculature', 'pulmonary', 'hemidiaphragms', 'lung', 'cardiomediastinal', 'pneumothorax', 'pleural']

class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value):
        # Cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.layernorm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(query)
        query = self.layernorm2(query + ffn_output)
        return query


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.d_model = args.d_model
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)

        keyword_token_size = self._keyword_token_size(tokenizer)
        self.image_proj = nn.Linear(args.d_vf, args.d_model)
        self.text_proj = nn.Linear(keyword_token_size, args.d_model)
     
        self.cross_attention = CrossAttention(d_model=args.d_model, n_heads=args.num_heads)
        self.out = nn.Linear(args.d_model, args.d_vf)

        self.register_buffer(
            'text_feature',
            self._build_keyword_features(text_list, tokenizer, keyword_token_size),
            persistent=False
        )
        self.register_buffer(
            'text_feature_mimic',
            self._build_keyword_features(text_list_mimic, tokenizer, keyword_token_size),
            persistent=False
        )

    @staticmethod
    def _keyword_token_size(tokenizer):
        return max(len(tokenizer(keyword)) for keyword in text_list + text_list_mimic)

    @staticmethod
    def _build_keyword_features(keywords, tokenizer, token_size):
        features = torch.zeros(len(keywords), token_size, dtype=torch.float32)
        for idx, keyword in enumerate(keywords):
            token_ids = torch.tensor(tokenizer(keyword), dtype=torch.float32)
            features[idx, :token_ids.numel()] = token_ids
        return features

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train', update_opts=None):
        if self.args.dataset_name == 'iu_xray':
            return self.forward_iu_xray(images, targets, mode, update_opts)
        return self.forward_mimic_cxr(images, targets, mode, update_opts)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts=None):
        update_opts = update_opts or {}
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        
        att_feats_0_proj = self.image_proj(att_feats_0)
        text_features = self.text_proj(self.text_feature)
        text_features = text_features.unsqueeze(0).expand(
            att_feats_0_proj.size(0),
            text_features.size(0),
            text_features.size(1)
        )

        att_feats_0 = self.cross_attention(
            query=att_feats_0_proj,
            key=text_features,
            value=text_features
        )
        att_feats_0 = self.out(att_feats_0)
        
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts=None):
        update_opts = update_opts or {}
        att_feats, fc_feats = self.visual_extractor(images)

        att_feats1 = self.image_proj(att_feats)
        text_features = self.text_proj(self.text_feature_mimic)
        text_features = text_features.unsqueeze(0).expand(
            att_feats1.size(0),
            text_features.size(0),
            text_features.size(1)
        )

        att_feats1 = self.cross_attention(
            query=att_feats1,
            key=text_features,
            value=text_features
        )
        att_feats1 = self.out(att_feats1)
        
        att_feats = torch.cat((att_feats, att_feats1), dim=1)
 
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError


BaseModel = BaseCMNModel
