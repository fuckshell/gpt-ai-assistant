"""Safe formula registry for curve fitting.

This module intentionally avoids `eval` and only exposes whitelisted
callables keyed by formula id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np

FormulaFn = Callable[..., np.ndarray]
FormulaSpec = dict


def uniform_linear_motion(t: np.ndarray, v: float, x0: float) -> np.ndarray:
    return v * t + x0


def uniform_acceleration(
    t: np.ndarray, a: float, v0: float, x0: float
) -> np.ndarray:
    return 0.5 * a * t**2 + v0 * t + x0


def simple_harmonic_motion(
    t: np.ndarray, A: float, omega: float, phi: float, offset: float
) -> np.ndarray:
    return A * np.cos(omega * t + phi) + offset


def damped_oscillation(
    t: np.ndarray, A: float, beta: float, omega: float, phi: float
) -> np.ndarray:
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


FORMULA_FUNCTIONS: Dict[str, FormulaFn] = {
    "uniform_linear_motion": uniform_linear_motion,
    "uniform_acceleration": uniform_acceleration,
    "simple_harmonic_motion": simple_harmonic_motion,
    "damped_oscillation": damped_oscillation,
}


def load_formula_catalog(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_formula_catalog(catalog: dict) -> List[FormulaSpec]:
    formulas: List[FormulaSpec] = []
    for group in catalog.values():
        if isinstance(group, list):
            formulas.extend(group)
    return formulas


def build_candidate_formulas(
    formula_catalog: Iterable[FormulaSpec], candidate_ids: Iterable[str]
) -> List[FormulaSpec]:
    wanted = set(candidate_ids)
    by_id = {formula["id"]: formula for formula in formula_catalog}
    return [by_id[item_id] for item_id in candidate_ids if item_id in wanted and item_id in by_id]

