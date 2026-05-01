import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # targets and masks should already be shifted by the caller ([:, 1:])
        target = target[:, :input.size(1)].to(input.device)
        mask = mask[:, :input.size(1)].to(input.device)
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / (torch.sum(mask) + 1e-10)
        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss

def compute_nll_sum_and_tokens(output, reports_ids, reports_masks):
    """
    Dùng cho validation/test aggregation chuẩn:
    trả về:
        nll_sum: tổng NLL trên token hợp lệ của batch
        token_count: tổng số token hợp lệ của batch
    """
    target = reports_ids
    mask = reports_masks

    target = target[:, :output.size(1)].to(output.device)
    mask = mask[:, :output.size(1)].float().to(output.device)

    nll = -output.gather(2, target.long().unsqueeze(2)).squeeze(2)
    nll = nll * mask

    nll_sum = nll.sum()
    token_count = mask.sum()

    return nll_sum, token_count