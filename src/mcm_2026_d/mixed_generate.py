from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .generate import generate_instance
from .schemas import Instance


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def _ensure_feasible_salary(vec: np.ndarray, cap: float, L: int) -> np.ndarray:
    vec = vec.astype(float)
    for _ in range(8):
        min_cost = float(np.sort(vec)[:L].sum())
        if min_cost <= cap:
            return vec
        factor = (cap / min_cost) * 0.98
        vec = vec * factor
        vec = np.clip(vec, 30_000.0, 260_000.0)
    return vec


def instance_from_real_pool(
    *,
    pool: pd.DataFrame,
    year_seq: list[int],
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
    """基于真实球员池构建一个小规模实例。

    pool 要求至少含：year, player, mp，并包含若干数值特征列。
    year_seq 长度必须为 T（每赛季用哪个真实年份的数据）。

    薪资：真实数据很难公开获得，这里用“分钟 + 胜利贡献代理(WS/BPM等)”构造代理薪资，
    再缩放确保满足工资帽并至少存在可行阵容。
    """

    if len(year_seq) != T:
        raise ValueError("year_seq length must equal T")

    rng = _rng(seed)

    # 先生成一个合成实例，借用其环境参数（cap, R_base, rho, Q_opp, w 等）
    base = generate_instance(seed=int(rng.integers(0, 2**31 - 1)), n_players=n_players, T=T, K=K, L=L, U=U)

    abilities: list[list[list[float]]] = []
    salaries: list[list[float]] = []

    # 从 pool 中选择可用数值列作为特征
    ignore = {"year", "player", "Tm", "tm", "mp", "g"}
    numeric_cols = [c for c in pool.columns if c not in ignore and pd.api.types.is_numeric_dtype(pool[c])]

    # 保证至少有 K-1 个“竞技特征”
    if len(numeric_cols) < max(1, K - 1):
        raise ValueError(f"Not enough numeric feature columns in pool: {numeric_cols}")

    feat_cols = numeric_cols[: max(1, K - 1)]

    for t, year in enumerate(year_seq):
        df_y = pool[pool["year"] == year].copy()
        if len(df_y) < n_players:
            raise ValueError(f"Not enough players for year={year}: {len(df_y)}")

        # 抽样 n_players 个球员
        df_s = df_y.sample(n=n_players, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))

        X = df_s[feat_cols].to_numpy(dtype=float)

        # 最后一维 marketability：用 mp +（如果存在）WS/BPM 的组合做一个代理
        mp = df_s.get("mp", pd.Series([0.0] * n_players)).to_numpy(dtype=float)
        ws = df_s.get("WS", pd.Series([0.0] * n_players)).to_numpy(dtype=float) if "WS" in df_s.columns else None
        bpm = df_s.get("BPM", pd.Series([0.0] * n_players)).to_numpy(dtype=float) if "BPM" in df_s.columns else None

        market = mp.copy()
        if ws is not None:
            market = market + 400.0 * ws
        if bpm is not None:
            market = market + 80.0 * bpm

        market = market.reshape(-1, 1)

        # 组合成 K 维能力（K-1 real + 1 market）
        if K == 1:
            A = _zscore(market)
        else:
            # 若 real 特征不足 K-1，重复/截断
            real = X
            if real.shape[1] < K - 1:
                reps = (K - 1 + real.shape[1] - 1) // real.shape[1]
                real = np.tile(real, reps)[:, : (K - 1)]
            else:
                real = real[:, : (K - 1)]

            A = np.concatenate([real, market], axis=1)
            A = _zscore(A)

        abilities.append(A.astype(float).tolist())

        # 薪资代理：base + 线性组合 + 噪声；再缩放确保可行
        sal = 70_000.0 + 0.22 * mp
        if ws is not None:
            sal = sal + 55_000.0 * ws
        if bpm is not None:
            sal = sal + 8_000.0 * bpm
        sal = sal + rng.normal(0.0, 15_000.0, size=n_players)
        sal = np.clip(sal, 30_000.0, 260_000.0)
        sal = _ensure_feasible_salary(sal, base.C[t], L)
        salaries.append(sal.astype(float).tolist())

    # 上赛季阵容：从第0赛季最便宜的 U 人构造一个可行 mask
    sal0 = np.array(salaries[0], dtype=float)
    order = np.argsort(sal0)
    mask = 0
    for i in order[:U]:
        mask |= 1 << int(i)

    return Instance(
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
        G=base.G,
        C=base.C,
        R_base=base.R_base,
        rho=base.rho,
        Q_opp=base.Q_opp,
        w=base.w,
        abilities=abilities,
        salaries=salaries,
        x_prev_mask=int(mask),
    )


def augment_real_instance(
    inst: Instance,
    *,
    seed: int | None = None,
    abilities_sigma: float = 0.10,
    salary_sigma: float = 0.05,
) -> Instance:
    """对一个“来自真实池”的实例做轻量数据增强。

    - abilities: 加高斯噪声后逐赛季做 z-score（保持尺度一致）
    - salaries: 乘性噪声 + clip + 重新缩放确保可行

    目标：不改变环境参数（C/R/Q/w 等），只让球员层面的输入更丰富。
    """

    rng = _rng(seed)

    A = np.asarray(inst.abilities, dtype=float)  # [T,n,K]
    A = A + rng.normal(0.0, abilities_sigma, size=A.shape)
    for t in range(inst.T):
        A[t] = _zscore(A[t])

    S = np.asarray(inst.salaries, dtype=float)  # [T,n]
    S = S * (1.0 + rng.normal(0.0, salary_sigma, size=S.shape))
    S = np.clip(S, 30_000.0, 260_000.0)
    for t in range(inst.T):
        S[t] = _ensure_feasible_salary(S[t], inst.C[t], inst.L)

    return Instance(
        id=str(uuid.uuid4()),
        n_players=inst.n_players,
        T=inst.T,
        K=inst.K,
        L=inst.L,
        U=inst.U,
        gamma=inst.gamma,
        lambda_win=inst.lambda_win,
        beta=inst.beta,
        churn_penalty=inst.churn_penalty,
        G=inst.G,
        C=inst.C,
        R_base=inst.R_base,
        rho=inst.rho,
        Q_opp=inst.Q_opp,
        w=inst.w,
        abilities=A.astype(float).tolist(),
        salaries=S.astype(float).tolist(),
        x_prev_mask=inst.x_prev_mask,
    )


def generate_real_augmented_instances(
    *,
    n: int,
    seed: int,
    pool: pd.DataFrame,
    years: list[int],
    aug_per_base: int = 2,
    abilities_sigma: float = 0.10,
    salary_sigma: float = 0.05,
    n_players: int = 15,
    T: int = 3,
    K: int = 6,
    L: int = 11,
    U: int = 12,
) -> list[tuple[Instance, dict]]:
    """生成 n 条样本，其中每条 base 实例配套若干条增强实例。

    约 1:aug_per_base 的增强比例（实际可能因截断略有偏差）。
    返回 (Instance, meta) 以便下游写入 JSONL 时标记来源。
    """

    if aug_per_base < 0:
        raise ValueError("aug_per_base must be >= 0")
    if n <= 0:
        return []

    rng = _rng(seed)
    out: list[tuple[Instance, dict]] = []

    while len(out) < n:
        s_base = int(rng.integers(0, 2**31 - 1))
        year_seq = [int(rng.choice(years)) for _ in range(T)]
        base = instance_from_real_pool(
            pool=pool,
            year_seq=year_seq,
            seed=s_base,
            n_players=n_players,
            T=T,
            K=K,
            L=L,
            U=U,
        )
        out.append(
            (
                base,
                {
                    "source": "real_pool",
                    "augmented": False,
                    "year_seq": year_seq,
                },
            )
        )
        if len(out) >= n:
            break

        for _ in range(aug_per_base):
            if len(out) >= n:
                break
            s_aug = int(rng.integers(0, 2**31 - 1))
            aug = augment_real_instance(
                base,
                seed=s_aug,
                abilities_sigma=abilities_sigma,
                salary_sigma=salary_sigma,
            )
            out.append(
                (
                    aug,
                    {
                        "source": "real_pool",
                        "augmented": True,
                        "year_seq": year_seq,
                        "abilities_sigma": float(abilities_sigma),
                        "salary_sigma": float(salary_sigma),
                    },
                )
            )

    return out


def generate_mixed_instances(
    *,
    n: int,
    seed: int,
    pool: pd.DataFrame | None,
    real_frac: float,
    years: list[int],
    n_players: int = 15,
    T: int = 3,
    K: int = 6,
    L: int = 11,
    U: int = 12,
) -> list[Instance]:
    rng = _rng(seed)
    out: list[Instance] = []

    for _ in range(n):
        use_real = pool is not None and (rng.random() < real_frac)
        s = int(rng.integers(0, 2**31 - 1))

        if use_real:
            assert pool is not None
            year_seq = [int(rng.choice(years)) for _ in range(T)]
            out.append(
                instance_from_real_pool(
                    pool=pool,
                    year_seq=year_seq,
                    seed=s,
                    n_players=n_players,
                    T=T,
                    K=K,
                    L=L,
                    U=U,
                )
            )
        else:
            out.append(generate_instance(seed=s, n_players=n_players, T=T, K=K, L=L, U=U))

    return out


def load_cached_pool(cache_csv_dir: Path) -> pd.DataFrame:
    """读取已缓存的 BRef 处理后球员池（由 CLI scrape-bref 产出）。"""

    files = sorted(cache_csv_dir.glob("pool_*.csv"))
    if not files:
        raise FileNotFoundError(f"No pool_*.csv under {cache_csv_dir}")

    dfs = [pd.read_csv(f) for f in files]
    pool = pd.concat(dfs, ignore_index=True)
    if "year" not in pool.columns:
        raise ValueError("cached pool missing year column")
    return pool
