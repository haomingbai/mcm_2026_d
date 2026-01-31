"""Neural training stack for MCM 2026 D.

This package implements a lightweight TFT-inspired policy/value model and
a supervised + actor-critic training pipeline.

Design goal: keep GPU memory < 8GB on typical consumer GPUs.
"""

from __future__ import annotations

