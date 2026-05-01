import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        self.is_densenet = self.visual_extractor.startswith('densenet')
        if self.is_densenet:
            self.model = model.features
        else:
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
        self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        patch_feats = self.model(images)
        if self.is_densenet:
            patch_feats = F.relu(patch_feats, inplace=True)
        avg_feats = self.avg_fnt(patch_feats).flatten(1)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

    """output:
    patch_feats: [B,H*W,2048] 
    avg_feats: [B,2048]"""