from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .math_utils import sigmoid
from .schemas import Instance, SeasonMetrics, Solution


@dataclass(frozen=True)
class PrecompSeason:
    masks: list[int]
    cost: dict[int, float]
    Q: dict[int, float]
    p: dict[int, float]
    W: dict[int, float]
    revenue: dict[int, float]
    profit: dict[int, float]
    base_reward: dict[int, float]  # 不含阵容变化惩罚


def _iter_masks(n_players: int) -> list[int]:
    return list(range(1 << n_players))


def _mask_size(mask: int) -> int:
    return mask.bit_count()


def _mask_to_indices(mask: int, n_players: int) -> list[int]:
    return [i for i in range(n_players) if (mask >> i) & 1]


def _compute_season_precomp(inst: Instance, t: int) -> PrecompSeason:
    n = inst.n_players
    L, U = inst.L, inst.U
    cap = inst.C[t]

    A = np.array(inst.abilities[t], dtype=float)  # (n,K)
    w = np.array(inst.w, dtype=float)  # (K,)
    sal = np.array(inst.salaries[t], dtype=float)  # (n,)

    masks: list[int] = []
    cost: dict[int, float] = {}
    Q: dict[int, float] = {}
    p: dict[int, float] = {}
    W: dict[int, float] = {}
    revenue: dict[int, float] = {}
    profit: dict[int, float] = {}
    base_reward: dict[int, float] = {}

    # 逐mask枚举（n<=15, 可接受）
    for mask in _iter_masks(n):
        sz = _mask_size(mask)
        if sz < L or sz > U:
            continue

        idx = _mask_to_indices(mask, n)
        c = float(sal[idx].sum())
        if c > cap:
            continue

        u = A[idx].sum(axis=0)
        q = float(w @ u)
        pt = sigmoid(inst.beta * (q - inst.Q_opp[t]))
        wt = float(inst.G[t] * pt)

        rev = float(inst.R_base[t] + inst.rho[t] * wt)
        prof = float(rev - c)

        # 按 model_general.md 的归一化写法
        reward = inst.lambda_win * (wt / float(inst.G[t])) + (1.0 - inst.lambda_win) * (prof / inst.R_base[t])

        masks.append(mask)
        cost[mask] = c
        Q[mask] = q
        p[mask] = pt
        W[mask] = wt
        revenue[mask] = rev
        profit[mask] = prof
        base_reward[mask] = reward

    if not masks:
        raise ValueError(f"No feasible roster found at season t={t}. Try relaxing L/U or cap.")

    return PrecompSeason(
        masks=masks,
        cost=cost,
        Q=Q,
        p=p,
        W=W,
        revenue=revenue,
        profit=profit,
        base_reward=base_reward,
    )


def solve_exact(inst: Instance) -> Solution:
    """精确求解：穷举可行阵容 + 备忘录DP，返回全局最优多赛季阵容序列。"""

    seasons = [_compute_season_precomp(inst, t) for t in range(inst.T)]

    @lru_cache(maxsize=None)
    def V(t: int, prev_mask: int) -> tuple[float, int]:
        # 返回 (最优值, 当前季最佳mask)
        if t >= inst.T:
            return 0.0, 0

        best_val = -1e100
        best_mask = 0
        season = seasons[t]

        # 遍历当季所有可行阵容
        for mask in season.masks:
            churn = (mask ^ prev_mask).bit_count()
            rt = season.base_reward[mask] - inst.churn_penalty * float(churn)
            val_next, _ = V(t + 1, mask)
            total = rt + inst.gamma * val_next
            if total > best_val:
                best_val = total
                best_mask = mask

        return best_val, best_mask

    # 回溯策略
    masks: list[int] = []
    per_season: list[SeasonMetrics] = []

    objective, first_mask = V(0, inst.x_prev_mask)
    prev = inst.x_prev_mask

    for t in range(inst.T):
        _, m = V(t, prev)
        masks.append(m)

        season = seasons[t]
        churn = (m ^ prev).bit_count()
        r = season.base_reward[m] - inst.churn_penalty * float(churn)

        per_season.append(
            SeasonMetrics(
                Q=season.Q[m],
                p=season.p[m],
                W=season.W[m],
                cost=season.cost[m],
                revenue=season.revenue[m],
                profit=season.profit[m],
                reward=r,
            )
        )

        prev = m

    rosters = [_mask_to_indices(m, inst.n_players) for m in masks]
    return Solution(id=inst.id, objective=float(objective), masks=masks, rosters=rosters, per_season=per_season)
