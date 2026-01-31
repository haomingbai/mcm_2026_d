from __future__ import annotations

import uuid
from dataclasses import replace

import numpy as np

from .schemas import Instance


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_instance(
    *,
    seed: int | None = None,
    n_players: int = 15,
    T: int = 3,
    K: int = 6,
    L: int = 11,
    U: int = 12,
    gamma: float = 0.9,
    lambda_win: float = 0.6,
    beta: float = 0.25,
    churn_penalty: float = 0.01,
) -> Instance:
    """生成一个“合理但合成”的小规模实例，确保可被精确穷举求解。

    设计原则：
    - 能力(标准化) + 市场/可用性等维度
    - 薪资与能力/市场性正相关
    - 工资帽与赛季参数接近 model_general.md 的数量级
    """

    rng = _rng(seed)

    # 赛季环境参数
    G = [44 for _ in range(T)]
    cap0 = rng.normal(1.55e6, 6.0e4)
    C = [float(cap0 * (1.03**t)) for t in range(T)]

    R0 = rng.normal(2.0e7, 2.0e6)
    R_base = [float(R0 * (1.02**t)) for t in range(T)]

    rho0 = rng.normal(6.0e5, 8.0e4)
    rho = [float(max(2.0e5, rho0) * (1.01**t)) for t in range(T)]

    Q_opp = [float(rng.normal(0.0, 0.5)) for _ in range(T)]

    # 能力权重（正向）
    w = rng.normal(0.0, 1.0, size=(K,))
    w = w / (np.linalg.norm(w) + 1e-9)

    # 生成球员静态“天赋”与随时间波动
    base_skill = rng.normal(0.0, 1.0, size=(n_players, K))
    # 将最后一维理解为“市场性/号召力”，让其略偏正并更稳定
    base_skill[:, -1] = rng.normal(0.3, 0.6, size=(n_players,))

    abilities: list[list[list[float]]] = []
    salaries: list[list[float]] = []

    def ensure_feasible_salary(vec: np.ndarray, cap: float, L_local: int) -> np.ndarray:
        # 通过缩放+截断，保证至少存在一个满足工资帽的L人阵容（取最便宜的L人）。
        # 这样精确求解阶段一定能找到可行解。
        vec = vec.astype(float)
        for _ in range(6):
            min_cost = float(np.sort(vec)[:L_local].sum())
            if min_cost <= cap:
                return vec
            factor = (cap / min_cost) * 0.98
            vec = vec * factor
            vec = np.clip(vec, 30_000.0, 260_000.0)
        return vec

    # 用一个年龄/成长趋势项做轻微时序变化
    trend = rng.normal(0.0, 0.15, size=(n_players, K))

    for t in range(T):
        noise = rng.normal(0.0, 0.25, size=(n_players, K))
        A_t = base_skill + t * trend + noise

        # 让每个维度近似标准化（z-score）
        A_t = (A_t - A_t.mean(axis=0)) / (A_t.std(axis=0) + 1e-9)
        abilities.append(A_t.astype(float).tolist())

        # 薪资：与(竞技能力)和(市场性)相关，且加噪声
        perf = A_t[:, :-1].mean(axis=1)
        market = A_t[:, -1]
        raw = 130_000 + 85_000 * perf + 105_000 * market + rng.normal(0.0, 25_000, size=n_players)
        raw = np.clip(raw, 30_000, 260_000)
        raw = ensure_feasible_salary(raw, C[t], L)
        salaries.append(raw.astype(float).tolist())

    # 给定一个上一赛季阵容：从第1赛季可行集合里随便抽一个（简单起见）
    # 为保证几乎必然存在可行阵容，先用贪婪选 U 人并尝试调整。
    sal0 = np.array(salaries[0], dtype=float)
    order = np.argsort(sal0)
    mask = 0
    # 先选最便宜的 U 个
    for i in order[:U]:
        mask |= 1 << int(i)

    # 如果仍超帽（通常不会），就再减少到 L
    if float((sal0 * np.array([(mask >> i) & 1 for i in range(n_players)], dtype=float)).sum()) > C[0]:
        mask = 0
        for i in order[:L]:
            mask |= 1 << int(i)

    inst = Instance(
        id=str(uuid.uuid4()),
        n_players=n_players,
        T=T,
        K=K,
        L=L,
        U=U,
        gamma=gamma,
        lambda_win=lambda_win,
        beta=beta,
        churn_penalty=churn_penalty,
        G=G,
        C=C,
        R_base=R_base,
        rho=rho,
        Q_opp=Q_opp,
        w=w.astype(float).tolist(),
        abilities=abilities,
        salaries=salaries,
        x_prev_mask=int(mask),
    )
    return inst


def generate_many(
    *,
    n: int,
    seed: int = 0,
    n_players: int = 15,
    T: int = 3,
    K: int = 6,
    L: int = 11,
    U: int = 12,
) -> list[Instance]:
    rng = _rng(seed)
    instances: list[Instance] = []
    for _ in range(n):
        s = int(rng.integers(0, 2**31 - 1))
        instances.append(
            generate_instance(
                seed=s,
                n_players=n_players,
                T=T,
                K=K,
                L=L,
                U=U,
            )
        )
    return instances
