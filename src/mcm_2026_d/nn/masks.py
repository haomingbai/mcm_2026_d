from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MaskSet:
    """Precomputed roster masks for n_players <= 15.

    We only include masks with size in [L, U]. Salary cap feasibility is handled
    per-season with a dynamic cost mask.
    """

    n_players: int
    L: int
    U: int
    masks: np.ndarray  # (M,) int64
    mask_matrix: np.ndarray  # (M, n_players) float32 in {0,1}
    sizes: np.ndarray  # (M,) int16

    def to_torch(self, device: torch.device) -> "TorchMaskSet":
        return TorchMaskSet(
            n_players=self.n_players,
            L=self.L,
            U=self.U,
            masks=torch.as_tensor(self.masks, dtype=torch.int64, device=device),
            mask_matrix=torch.as_tensor(self.mask_matrix, dtype=torch.float32, device=device),
            sizes=torch.as_tensor(self.sizes, dtype=torch.float32, device=device),
        )


@dataclass(frozen=True)
class TorchMaskSet:
    n_players: int
    L: int
    U: int
    masks: torch.Tensor  # (M,) int64
    mask_matrix: torch.Tensor  # (M,n) float32
    sizes: torch.Tensor  # (M,) float32


def build_mask_set(*, n_players: int, L: int, U: int) -> MaskSet:
    if n_players <= 0 or n_players > 15:
        raise ValueError("n_players must be in [1, 15]")
    if L < 0 or U < 0 or L > U:
        raise ValueError("invalid L/U")

    masks: list[int] = []
    sizes: list[int] = []

    for m in range(1 << n_players):
        sz = int(m.bit_count())
        if sz < L or sz > U:
            continue
        masks.append(m)
        sizes.append(sz)

    masks_arr = np.asarray(masks, dtype=np.int64)
    sizes_arr = np.asarray(sizes, dtype=np.int16)

    # (M,n) binary matrix
    M = masks_arr.shape[0]
    mat = np.zeros((M, n_players), dtype=np.float32)
    for i in range(n_players):
        mat[:, i] = ((masks_arr >> i) & 1).astype(np.float32)

    return MaskSet(
        n_players=n_players,
        L=L,
        U=U,
        masks=masks_arr,
        mask_matrix=mat,
        sizes=sizes_arr,
    )


def build_popcount_table(*, n_players: int) -> np.ndarray:
    """Returns popcount for all 2^n masks."""

    if n_players <= 0 or n_players > 15:
        raise ValueError("n_players must be in [1, 15]")
    size = 1 << n_players
    out = np.zeros((size,), dtype=np.int16)
    for m in range(size):
        out[m] = int(m.bit_count())
    return out
