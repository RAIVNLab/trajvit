from scipy.optimize import linear_sum_assignment  # CPU (M is tiny)
import torch


def hungarian_per_sample(
    logits: torch.Tensor,     # (N,M)  predicted logits for one video
    labels: torch.Tensor,     # (N)    GT indices 0..M-1 or -1
    valid_gt: torch.Tensor,   # (M)    which GT trajs are real
) -> torch.LongTensor:        # permutation index p s.t. p[g] = matched pred
    """
    Returns a permutation tensor `perm` of length M.
    Re-indexing with `perm` aligns predictions with GT trajectories.
    """
    device = logits.device
    M = logits.size(1)

    # ---- 1.  build cost matrix (M_gt × M_pred)
    # Here we use  −#correct_pixels  (negative because Hungarian minimises)
    preds = logits.argmax(-1)                          # (N)
    cost = torch.full((M, M), 0., device=device)       # float
    for g in torch.nonzero(valid_gt, as_tuple=False).flatten():
        mask_g = labels.eq(g)
        if mask_g.any():
            # counts[g, m] = how many of GT-traj g’s pixels the model
            #          currently puts in predicted cluster m
            counts = torch.bincount(preds[mask_g], minlength=M).float()
            cost[g] = -counts                          # bigger counts → lower cost

    # ---- 2.  Hungarian on CPU (M is ≤ ~128, so overhead is tiny)
    row, col = linear_sum_assignment(cost.cpu().numpy())
    perm = torch.arange(M)
    perm[row] = torch.tensor(col)                      # perm[g] = best pred
    return perm.to(device)