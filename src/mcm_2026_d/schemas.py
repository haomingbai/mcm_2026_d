from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Instance:
    id: str
    n_players: int
    T: int
    K: int
    L: int
    U: int

    gamma: float
    lambda_win: float
    beta: float
    churn_penalty: float

    G: list[int]
    C: list[float]
    R_base: list[float]
    rho: list[float]
    Q_opp: list[float]

    w: list[float]
    abilities: list[list[list[float]]]  # [T][n_players][K]
    salaries: list[list[float]]  # [T][n_players]

    x_prev_mask: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeasonMetrics:
    Q: float
    p: float
    W: float
    cost: float
    revenue: float
    profit: float
    reward: float


@dataclass(frozen=True)
class Solution:
    id: str
    objective: float
    masks: list[int]
    rosters: list[list[int]]
    per_season: list[SeasonMetrics]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_season"] = [asdict(m) for m in self.per_season]
        return d
