"""Project Newton runtime configuration."""

from __future__ import annotations

# Global scale factor set after manual calibration in perception stage.
PIXELS_PER_METER: float = 0.0

# Train/test split ratio for extrapolation validation.
SPLIT_RATIO: float = 0.8

# Selection defaults used by fitting module.
AIC_WEIGHT: float = 0.2
TEST_MSE_THRESHOLD: float = 1e-3

