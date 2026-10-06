"""Time-ordered forecast evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForecastObservation:
    """One probability forecast and its realized binary outcome."""

    probability: float
    outcome: int


def brier_score(observations: list[ForecastObservation]) -> float:
    """Calculate mean squared probability error."""
    if not observations:
        raise ValueError("at least one forecast observation is required")
    return sum((item.probability - item.outcome) ** 2 for item in observations) / len(observations)


def calibration_bins(
    observations: list[ForecastObservation],
    bin_count: int = 5,
) -> list[dict[str, Any]]:
    """Return forecast-frequency calibration bins."""
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = [
            item
            for item in observations
            if lower <= item.probability < upper or (index == bin_count - 1 and item.probability == 1)
        ]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_probability": sum(item.probability for item in members) / len(members)
                if members
                else None,
                "observed_frequency": sum(item.outcome for item in members) / len(members)
                if members
                else None,
            }
        )
    return bins
