import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
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
    target = reports_ids[:, 1:]
    mask = reports_masks[:, 1:]

    target = target[:, :output.size(1)]
    mask = mask[:, :output.size(1)].float()

    nll = -output.gather(2, target.long().unsqueeze(2)).squeeze(2)
    nll = nll * mask

    nll_sum = nll.sum()
    token_count = mask.sum()

    return nll_sum, token_count