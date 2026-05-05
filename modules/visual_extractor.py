import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.is_swin = self.visual_extractor.startswith('swin')

        model = self._build_backbone()
        if self.is_swin:
            self.model = nn.Sequential(model.features, model.norm)
            self.avg_fnt = None
        else:
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _build_backbone(self):
        constructor = getattr(models, self.visual_extractor)
        try:
            return constructor(pretrained=self.pretrained)
        except TypeError:
            weights = 'DEFAULT' if self.pretrained else None
            return constructor(weights=weights)

    def forward(self, images):
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
    """
