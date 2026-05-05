import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.output_dim = args.d_vf
        self.visual_fusion = getattr(args, 'visual_fusion', 'gated')
        self.is_ensemble = self.visual_extractor in (
            'resnet101_swin_t',
            'resnet_swin_t',
            'ensemble_resnet101_swin_t',
        )
        self.is_swin = self.visual_extractor.startswith('swin')

        if self.is_ensemble:
            resnet = self._build_backbone('resnet101')
            swin = self._build_backbone('swin_t')
            self.resnet_model = nn.Sequential(*list(resnet.children())[:-2])
            self.swin_model = nn.Sequential(swin.features, swin.norm)
            self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))
            if self.visual_fusion == 'concat':
                self.fusion_proj = nn.Linear(2048 + 768, self.output_dim)
            elif self.visual_fusion == 'gated':
                self.resnet_proj = nn.Linear(2048, self.output_dim)
                self.swin_proj = nn.Linear(768, self.output_dim)
                self.fusion_gate = nn.Linear(self.output_dim * 2, 1)
            else:
                raise ValueError(f'Unsupported visual_fusion: {self.visual_fusion}')
            return

        model = self._build_backbone(self.visual_extractor)
        if self.is_swin:
            self.model = nn.Sequential(model.features, model.norm)
            self.avg_fnt = None
        else:
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _build_backbone(self, model_name):
        constructor = getattr(models, model_name)
        try:
            return constructor(pretrained=self.pretrained)
        except TypeError:
            weights = 'DEFAULT' if self.pretrained else None
            return constructor(weights=weights)

    def _forward_resnet(self, images):
        patch_feats = self.resnet_model(images)
        avg_feats = self.avg_fnt(patch_feats).flatten(1)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

    def _forward_swin(self, images):
        patch_feats = self.swin_model(images)
        batch_size, height, width, feat_size = patch_feats.shape
        avg_feats = patch_feats.mean(dim=(1, 2))
        patch_feats = patch_feats.reshape(batch_size, height * width, feat_size)
        return patch_feats, avg_feats

    def forward(self, images):
        if self.is_ensemble:
            resnet_patch_feats, resnet_avg_feats = self._forward_resnet(images)
            swin_patch_feats, swin_avg_feats = self._forward_swin(images)

            if self.visual_fusion == 'concat':
                patch_feats = torch.cat((resnet_patch_feats, swin_patch_feats), dim=-1)
                avg_feats = torch.cat((resnet_avg_feats, swin_avg_feats), dim=-1)
                return self.fusion_proj(patch_feats), self.fusion_proj(avg_feats)

            resnet_patch_feats = self.resnet_proj(resnet_patch_feats)
            swin_patch_feats = self.swin_proj(swin_patch_feats)
            patch_gate = torch.sigmoid(
                self.fusion_gate(torch.cat((resnet_patch_feats, swin_patch_feats), dim=-1))
            )
            patch_feats = patch_gate * resnet_patch_feats + (1 - patch_gate) * swin_patch_feats

            resnet_avg_feats = self.resnet_proj(resnet_avg_feats)
            swin_avg_feats = self.swin_proj(swin_avg_feats)
            avg_gate = torch.sigmoid(
                self.fusion_gate(torch.cat((resnet_avg_feats, swin_avg_feats), dim=-1))
            )
            avg_feats = avg_gate * resnet_avg_feats + (1 - avg_gate) * swin_avg_feats
            return patch_feats, avg_feats

        patch_feats = self.model(images)

        if self.is_swin:
            # torchvision Swin keeps features as [B, H, W, C].
            batch_size, height, width, feat_size = patch_feats.shape
            avg_feats = patch_feats.mean(dim=(1, 2))
            patch_feats = patch_feats.reshape(batch_size, height * width, feat_size)
            return patch_feats, avg_feats

        avg_feats = self.avg_fnt(patch_feats).flatten(1)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

    """output:
    resnet101:
        patch_feats: [B,H*W,2048]
        avg_feats: [B,2048]
    swin_t / swin_s:
        patch_feats: [B,H*W,768]
        avg_feats: [B,768]
    swin_b:
        patch_feats: [B,H*W,1024]
        avg_feats: [B,1024]
    resnet101_swin_t:
        concat: [2048+768] -> project to args.d_vf
        gated: resnet/swin -> args.d_vf, then patch-wise scalar gate
    """
