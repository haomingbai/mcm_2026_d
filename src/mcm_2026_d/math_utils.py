from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    # 数值稳定的 sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)
