from __future__ import annotations

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def compute_season_reward_all_actions(
    *,
    mask_matrix: torch.Tensor,  # (M,n) float
    sizes: torch.Tensor,  # (M,) float
    masks: torch.Tensor,  # (M,) int
    popcount_table: torch.Tensor,  # (2^n,) int
    abilities_t: torch.Tensor,  # (B,n,K)
    salaries_t: torch.Tensor,  # (B,n)
    w: torch.Tensor,  # (B,K)
    G_t: torch.Tensor,  # (B,)
    C_t: torch.Tensor,  # (B,)
    R_base_t: torch.Tensor,  # (B,)
    rho_t: torch.Tensor,  # (B,)
    Q_opp_t: torch.Tensor,  # (B,)
    beta: torch.Tensor,  # (B,)
    lambda_win: torch.Tensor,  # (B,)
    churn_penalty: torch.Tensor,  # (B,)
    prev_mask: torch.Tensor,  # (B,) int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (reward_all, feasible_mask, cost_all, churn_all).

    Shapes:
      - reward_all: (B,M)
      - feasible_mask: (B,M) bool (cap-feasible)
      - cost_all: (B,M)
      - churn_all: (B,M)

    Note: mask size feasibility is already guaranteed by how mask_matrix is built.
    """

    B, n, K = abilities_t.shape
    M = mask_matrix.shape[0]

    # cost: (B,M) = salaries (B,n) with mask_matrix (M,n)
    cost_all = torch.einsum("mn,bn->bm", mask_matrix, salaries_t)
    feasible = cost_all <= C_t[:, None]

    # sum abilities: (B,M,K)
    sum_abil = torch.einsum("mn,bnk->bmk", mask_matrix, abilities_t)
    q_all = torch.einsum("bmk,bk->bm", sum_abil, w)

    p_all = sigmoid(beta[:, None] * (q_all - Q_opp_t[:, None]))
    W_all = G_t[:, None] * p_all

    revenue_all = R_base_t[:, None] + rho_t[:, None] * W_all
    profit_all = revenue_all - cost_all

    base_reward = lambda_win[:, None] * (W_all / (G_t[:, None] + 1e-9)) + (1.0 - lambda_win[:, None]) * (
        profit_all / (R_base_t[:, None] + 1e-9)
    )

    # churn_all via popcount table on XOR
    xor = torch.bitwise_xor(masks[None, :].to(prev_mask.dtype), prev_mask[:, None])
    churn_all = popcount_table[xor]

    reward_all = base_reward - churn_penalty[:, None] * churn_all.to(base_reward.dtype)

    # Infeasible actions should never be selected; mask downstream with -inf logits.
    return reward_all, feasible, cost_all, churn_all
