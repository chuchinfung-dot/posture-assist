"""EMA smoother for noisy geometric metrics."""

from __future__ import annotations

from typing import Dict


class EmaSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._values: Dict[str, float] = {}

    def update(self, key: str, value: float) -> float:
        if key not in self._values:
            self._values[key] = value
        else:
            a = self.alpha
            self._values[key] = a * value + (1.0 - a) * self._values[key]
        return self._values[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self._values.get(key, default)

    def reset(self) -> None:
        self._values.clear()
