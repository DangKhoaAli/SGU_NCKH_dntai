import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, weights=None):
        # truncate to the same size
        target = target[:, :input.size(1)].to(input.device)
        mask = mask[:, :input.size(1)].float().to(input.device)

        nll = -input.gather(2, target.long().unsqueeze(2)).squeeze(2)

        if weights is None:
            effective_mask = mask
        else:
            weights = weights[:, :input.size(1)].float().to(input.device)
            effective_mask = mask * weights

        loss = torch.sum(nll * effective_mask) / torch.sum(effective_mask).clamp_min(1e-8)
        return loss


def compute_loss(output, reports_ids, reports_masks, reports_weights=None):
    criterion = LanguageModelCriterion()
    weights = reports_weights[:, 1:] if reports_weights is not None else None
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:], weights).mean()
    return loss

def compute_nll_sum_and_tokens(output, reports_ids, reports_masks, reports_weights=None):
    """
    Dùng cho validation/test aggregation chuẩn:
    trả về:
        nll_sum: tổng NLL trên token hợp lệ của batch, có weight nếu reports_weights được truyền
        token_count: tổng mask/weight hợp lệ của batch
    """
    target = reports_ids[:, 1:].to(output.device)
    mask = reports_masks[:, 1:].float().to(output.device)
    weights = reports_weights[:, 1:] if reports_weights is not None else None

    target = target[:, :output.size(1)]
    mask = mask[:, :output.size(1)]
    if weights is None:
        effective_mask = mask
    else:
        weights = weights[:, :output.size(1)].float().to(output.device)
        effective_mask = mask * weights

    nll = -output.gather(2, target.long().unsqueeze(2)).squeeze(2)
    nll = nll * effective_mask

    nll_sum = nll.sum()
    token_count = effective_mask.sum().clamp_min(1e-8)

    return nll_sum, token_count
