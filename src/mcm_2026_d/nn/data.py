from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Batch:
    ids: tuple[str, ...]
    abilities: torch.Tensor  # (B,T,n,K) float32
    salaries: torch.Tensor  # (B,T,n) float32
    env: torch.Tensor  # (B,T,E) float32
    w: torch.Tensor  # (B,K) float32
    x_prev_mask: torch.Tensor  # (B,) int64
    teacher_masks: torch.Tensor  # (B,T) int64
    gamma: torch.Tensor  # (B,) float32
    lambda_win: torch.Tensor  # (B,) float32
    beta: torch.Tensor  # (B,) float32
    churn_penalty: torch.Tensor  # (B,) float32


class PairDataset(Dataset):
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"Empty dataset: {self.path}")

        # Basic schema sanity
        inst0 = self.rows[0]["instance"]
        self.n_players = int(inst0["n_players"])
        self.T = int(inst0["T"])
        self.K = int(inst0["K"])
        self.L = int(inst0["L"])
        self.U = int(inst0["U"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_pairs(batch: list[dict[str, Any]]) -> Batch:
    insts = [b["instance"] for b in batch]
    sols = [b["solution"] for b in batch]

    ids = tuple(str(inst["id"]) for inst in insts)

    B = len(batch)
    T = int(insts[0]["T"])
    n = int(insts[0]["n_players"])
    K = int(insts[0]["K"])

    abilities = torch.zeros((B, T, n, K), dtype=torch.float32)
    salaries = torch.zeros((B, T, n), dtype=torch.float32)
    w = torch.zeros((B, K), dtype=torch.float32)

    # env per season: [G, C, R_base, rho, Q_opp] + season index (scaled)
    E = 6
    env = torch.zeros((B, T, E), dtype=torch.float32)

    x_prev_mask = torch.zeros((B,), dtype=torch.int64)
    teacher_masks = torch.zeros((B, T), dtype=torch.int64)

    gamma = torch.zeros((B,), dtype=torch.float32)
    lambda_win = torch.zeros((B,), dtype=torch.float32)
    beta = torch.zeros((B,), dtype=torch.float32)
    churn_penalty = torch.zeros((B,), dtype=torch.float32)

    for i, inst in enumerate(insts):
        abilities[i] = torch.as_tensor(inst["abilities"], dtype=torch.float32)
        salaries[i] = torch.as_tensor(inst["salaries"], dtype=torch.float32)
        w[i] = torch.as_tensor(inst["w"], dtype=torch.float32)

        x_prev_mask[i] = int(inst["x_prev_mask"])
        teacher_masks[i] = torch.as_tensor(sols[i]["masks"], dtype=torch.int64)

        gamma[i] = float(inst["gamma"])
        lambda_win[i] = float(inst["lambda_win"])
        beta[i] = float(inst["beta"])
        churn_penalty[i] = float(inst["churn_penalty"])

        G = np.asarray(inst["G"], dtype=np.float32)
        C = np.asarray(inst["C"], dtype=np.float32)
        R_base = np.asarray(inst["R_base"], dtype=np.float32)
        rho = np.asarray(inst["rho"], dtype=np.float32)
        Q_opp = np.asarray(inst["Q_opp"], dtype=np.float32)

        # Normalize to tame scales for NN
        # - C: ~1.6e6
        # - R_base: ~2e7
        # - rho: ~6e5
        env_i = np.stack(
            [
                G / 50.0,
                C / 2.0e6,
                R_base / 2.5e7,
                rho / 8.0e5,
                Q_opp / 3.0,
                (np.arange(T, dtype=np.float32) / max(1, (T - 1))).astype(np.float32),
            ],
            axis=-1,
        )
        env[i] = torch.as_tensor(env_i, dtype=torch.float32)

    return Batch(
        ids=ids,
        abilities=abilities,
        salaries=salaries,
        env=env,
        w=w,
        x_prev_mask=x_prev_mask,
        teacher_masks=teacher_masks,
        gamma=gamma,
        lambda_win=lambda_win,
        beta=beta,
        churn_penalty=churn_penalty,
    )
