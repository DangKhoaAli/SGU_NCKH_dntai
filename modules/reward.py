import numpy as np
import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

def get_self_critical_reward(greedy_res, sample_res, gts_ids, tokenizer, reward_type='cider'):
    """
    Compute SCST rewards for a batch.
    
    greedy_res: (B, L)
    sample_res: (B, L)
    gts_ids: (B, L) - ground truth with <bos>
    """
    batch_size = greedy_res.shape[0]
    
    # Decode to strings
    res_greedy = tokenizer.decode_batch(greedy_res.cpu().numpy())
    res_sample = tokenizer.decode_batch(sample_res.cpu().numpy())
    # gts_ids usually contains <bos> at index 0, so we skip it if tokenizer.decode expects clean tokens
    # In this codebase, decode_batch handles the tokens.
    gts = tokenizer.decode_batch(gts_ids[:, 1:].cpu().numpy())

    # Prepare for pycocoevalcap format: {id: [string]}
    res_greedy_dict = {i: [res_greedy[i]] for i in range(batch_size)}
    res_sample_dict = {i: [res_sample[i]] for i in range(batch_size)}
    gts_dict = {i: [gts[i]] for i in range(batch_size)}

    if reward_type == 'cider':
        scorer = Cider()
        _, scores_greedy = scorer.compute_score(gts_dict, res_greedy_dict)
        _, scores_sample = scorer.compute_score(gts_dict, res_sample_dict)
    elif reward_type == 'bleu':
        scorer = Bleu(4)
        score_greedy, scores_greedy = scorer.compute_score(gts_dict, res_greedy_dict)
        score_sample, scores_sample = scorer.compute_score(gts_dict, res_sample_dict)
        # scores_greedy is a list of 4 arrays, use BLEU-4 (last one)
        scores_greedy = scores_greedy[3]
        scores_sample = scores_sample[3]
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    # Reward = R(sample) - R(greedy)
    rewards = scores_sample - scores_greedy
    
    return torch.from_numpy(rewards).float(), torch.from_numpy(scores_sample).float()
