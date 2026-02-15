"""Lightweight perception helpers for Newton MVP.

This module keeps v1 intentionally simple:
- get calibration (pixels per meter)
- convert pixel trajectory into metric trajectory
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

import config


def parse_length_to_meters(length_text: str) -> float:
    """Parse text like '0.3m', '30cm', '120mm' into meters."""
    text = length_text.strip().lower()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(m|cm|mm)", text)
    if not match:
        raise ValueError(f"Unsupported length format: '{length_text}'")
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return value
    if unit == "cm":
        return value / 100.0
    return value / 1000.0


def calibrate_pixels_per_meter(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    real_distance_m: float,
) -> float:
    if real_distance_m <= 0:
        raise ValueError("real_distance_m must be positive")
    pixel_distance = math.dist(point_a, point_b)
    if pixel_distance <= 0:
        raise ValueError("Calibration points are identical")
    return pixel_distance / real_distance_m


def interactive_calibration() -> float:
    print("Enter ruler endpoint A as 'x,y':")
    ax, ay = [float(item.strip()) for item in input().split(",")]
    print("Enter ruler endpoint B as 'x,y':")
    bx, by = [float(item.strip()) for item in input().split(",")]
    print("Enter real distance (e.g. 0.3m, 30cm):")
    real_distance_m = parse_length_to_meters(input())
    ppm = calibrate_pixels_per_meter((ax, ay), (bx, by), real_distance_m)
    print(f"pixels_per_meter={ppm:.6f}")
    return ppm


def pixels_to_meters(values: Iterable[float], pixels_per_meter: float) -> np.ndarray:
    if pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be positive")
    return np.asarray(list(values), dtype=float) / pixels_per_meter


def convert_pixel_csv_to_meter_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    pixels_per_meter: float | None = None,
) -> None:
    """Convert CSV columns x_pix/y_pix into x_meter/y_meter."""
    ppm = pixels_per_meter if pixels_per_meter is not None else config.PIXELS_PER_METER
    if ppm <= 0:
        raise ValueError(
            "PIXELS_PER_METER is not calibrated. Set config or pass pixels_per_meter."
        )

    input_path = Path(input_csv)
    output_path = Path(output_csv)
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_pix = float(row["x_pix"])
            y_pix = float(row["y_pix"])
            rows.append(
                {
                    "time": float(row["time"]),
                    "x_meter": x_pix / ppm,
                    "y_meter": y_pix / ppm,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "x_meter", "y_meter"])
        writer.writeheader()
        writer.writerows(rows)

