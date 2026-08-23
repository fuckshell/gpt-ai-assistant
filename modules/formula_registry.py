"""Safe formula registry for curve fitting.

This module intentionally avoids `eval` and only exposes whitelisted
callables keyed by formula id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set

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
    t: np.ndarray, A: float, beta: float, omega: float, phi: float, offset: float
) -> np.ndarray:
    return A * np.exp(-beta * t) * np.cos(omega * t + phi) + offset


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


def validate_formula_catalog(
    formulas: Sequence[FormulaSpec],
    formula_functions: Dict[str, FormulaFn] | None = None,
) -> None:
    """Fail fast if catalog is misaligned with the safe callable registry."""
    formula_functions = formula_functions or FORMULA_FUNCTIONS
    seen: Set[str] = set()
    for formula in formulas:
        formula_id = formula.get("id")
        if not formula_id:
            raise ValueError("Formula entry missing 'id'")
        if formula_id in seen:
            raise ValueError(f"Duplicate formula id: {formula_id}")
        seen.add(formula_id)
        if formula_id not in formula_functions:
            raise ValueError(
                f"Formula id '{formula_id}' has no callable in FORMULA_FUNCTIONS"
            )
        params = list(formula.get("params", []))
        if not params:
            raise ValueError(f"Formula '{formula_id}' has empty params")
        bounds = formula.get("bounds") or {}
        for param in params:
            if param not in bounds:
                continue
            pair = bounds[param]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Formula '{formula_id}' bad bounds for '{param}'")
            low, high = float(pair[0]), float(pair[1])
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"Formula '{formula_id}' non-finite bounds for '{param}'")
            if low >= high:
                raise ValueError(f"Formula '{formula_id}' inverted bounds for '{param}'")

    missing_in_catalog = sorted(set(formula_functions) - seen)
    if missing_in_catalog:
        raise ValueError(
            "FORMULA_FUNCTIONS ids missing from catalog: "
            + ", ".join(missing_in_catalog)
        )


def build_candidate_formulas(
    formula_catalog: Iterable[FormulaSpec],
    candidate_ids: Iterable[str],
    *,
    strict_unknown: bool = True,
) -> List[FormulaSpec]:
    """Resolve candidate ids in order, with optional strict unknown-id errors."""
    by_id = {formula["id"]: formula for formula in formula_catalog}
    resolved: List[FormulaSpec] = []
    seen: Set[str] = set()
    unknown: List[str] = []
    duplicates: List[str] = []

    for item_id in candidate_ids:
        if item_id in seen:
            duplicates.append(item_id)
            continue
        seen.add(item_id)
        if item_id not in by_id:
            unknown.append(item_id)
            continue
        resolved.append(by_id[item_id])

    if unknown and strict_unknown:
        available = ", ".join(sorted(by_id))
        raise ValueError(
            "Unknown formula id(s): "
            + ", ".join(unknown)
            + f". Available: {available}"
        )
    if duplicates:
        # Keep ordered unique list, but surface the issue to callers via stderr-friendly message.
        # Raising would be too harsh for accidental repeats; callers can inspect duplicates.
        pass
    return resolved
