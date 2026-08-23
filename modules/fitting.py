"""Model fitting and extrapolation-based model selection for Newton v3.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

from config import AIC_WEIGHT, TEST_MSE_THRESHOLD
from modules.formula_registry import FORMULA_FUNCTIONS, FormulaFn
from modules.json_util import json_safe

EPS = 1e-12


@dataclass
class FitResult:
    formula_id: str
    success: bool
    params: Dict[str, float]
    mse_train: float
    mse_test: float
    aic: float
    aicc: float
    score: float
    passed: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


def validate_series(
    t_data: Sequence[float],
    x_data: Sequence[float],
    *,
    require_strictly_increasing_time: bool = True,
    min_points: int = 1,
    name: str = "series",
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate 1D equal-length finite arrays; optionally enforce time order."""
    t = np.asarray(t_data, dtype=float)
    x = np.asarray(x_data, dtype=float)
    if t.ndim != 1 or x.ndim != 1:
        raise ValueError(f"{name} must be 1D arrays")
    if t.shape != x.shape:
        raise ValueError(f"{name} t and x must have identical shapes")
    if t.size < min_points:
        raise ValueError(f"{name} needs at least {min_points} points, got {t.size}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf values")
    if require_strictly_increasing_time:
        diffs = np.diff(t)
        if np.any(diffs <= 0):
            raise ValueError(
                f"{name} time must be strictly increasing (found duplicates or reversals)"
            )
    return t, x


def split_train_test_time_series(
    t_data: Sequence[float],
    x_data: Sequence[float],
    split_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    t, x = validate_series(
        t_data,
        x_data,
        require_strictly_increasing_time=True,
        min_points=6,
        name="trajectory",
    )
    cut = max(2, min(t.size - 2, int(round(t.size * split_ratio))))
    return t[:cut], x[:cut], t[cut:], x[cut:]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def compute_aic(mse_train: float, sample_size: int, num_params: int) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    return float((2 * num_params) + sample_size * np.log(mse_train + EPS))


def compute_aicc(aic: float, sample_size: int, num_params: int) -> float:
    denom = sample_size - num_params - 1
    if denom <= 0:
        return float("inf")
    correction = (2 * num_params * (num_params + 1)) / denom
    return float(aic + correction)


def _build_bounds(params: List[str], bounds_cfg: dict | None) -> Tuple[np.ndarray, np.ndarray]:
    if not bounds_cfg:
        size = len(params)
        return np.full(size, -np.inf), np.full(size, np.inf)
    lower, upper = [], []
    for param in params:
        low, high = bounds_cfg.get(param, (-np.inf, np.inf))
        lower.append(low)
        upper.append(high)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _clip_guess(guess: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    clipped = guess.astype(float).copy()
    for idx, (low, high) in enumerate(zip(lower, upper)):
        if np.isfinite(low) and np.isfinite(high):
            span = high - low
            clipped[idx] = float(np.clip(clipped[idx], low + 1e-9 * span, high - 1e-9 * span))
        elif np.isfinite(low):
            clipped[idx] = max(float(clipped[idx]), float(low) + 1e-9)
        elif np.isfinite(high):
            clipped[idx] = min(float(clipped[idx]), float(high) - 1e-9)
    return clipped


def _midpoint_guess(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    guesses = []
    for low, high in zip(lower, upper):
        if np.isfinite(low) and np.isfinite(high):
            guesses.append((low + high) / 2.0)
        elif np.isfinite(low):
            guesses.append(low + 1.0)
        elif np.isfinite(high):
            guesses.append(high - 1.0)
        else:
            guesses.append(0.0)
    return np.asarray(guesses, dtype=float)


def _estimate_dominant_omega(t: np.ndarray, x: np.ndarray) -> float:
    """Estimate angular frequency from FFT peak of a detrended signal."""
    if t.size < 8:
        return 1.0
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return 1.0
    signal = x - np.mean(x)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    if freqs.size <= 1:
        return 1.0
    # Ignore DC bin.
    power = np.abs(spectrum[1:])
    peak_idx = int(np.argmax(power)) + 1
    freq = float(freqs[peak_idx])
    if freq <= 0:
        return 1.0
    return 2.0 * np.pi * freq


def _oscillation_seeds(
    formula_id: str,
    param_names: List[str],
    t_train: np.ndarray,
    x_train: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> List[np.ndarray]:
    offset = float(np.mean(x_train))
    amp = float(0.5 * (np.max(x_train) - np.min(x_train)))
    amp = max(amp, 1e-3)
    omega = _estimate_dominant_omega(t_train, x_train)
    omega = float(np.clip(omega, 0.05, 40.0))
    phases = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    omega_grid = [omega, 0.5 * omega, 2.0 * omega, max(omega / 3.0, 0.1)]
    betas = [0.0, 0.2, 1.0] if formula_id == "damped_oscillation" else [None]

    seeds: List[np.ndarray] = []
    for omega_guess in omega_grid:
        for phi in phases:
            for beta in betas:
                values = {
                    "A": amp,
                    "omega": omega_guess,
                    "phi": float(phi),
                    "offset": offset,
                    "beta": 0.0 if beta is None else beta,
                }
                guess = np.asarray([values.get(name, 0.0) for name in param_names], dtype=float)
                seeds.append(_clip_guess(guess, lower, upper))

    seeds.append(_clip_guess(_midpoint_guess(lower, upper), lower, upper))
    # Deduplicate roughly.
    unique: List[np.ndarray] = []
    for seed in seeds:
        if any(np.allclose(seed, existing, rtol=1e-3, atol=1e-6) for existing in unique):
            continue
        unique.append(seed)
    return unique


def _initial_guesses(
    formula: dict,
    t_train: np.ndarray,
    x_train: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> List[np.ndarray]:
    formula_id = formula["id"]
    param_names = list(formula.get("params", []))
    if formula_id in {"simple_harmonic_motion", "damped_oscillation"}:
        return _oscillation_seeds(formula_id, param_names, t_train, x_train, lower, upper)

    # Kinematic models: use linear/quadratic regression inspired seeds + midpoint.
    seeds = [_clip_guess(_midpoint_guess(lower, upper), lower, upper)]
    if formula_id == "uniform_linear_motion" and set(param_names) == {"v", "x0"}:
        v, x0 = np.polyfit(t_train, x_train, 1)
        seeds.insert(0, _clip_guess(np.asarray([v, x0], dtype=float), lower, upper))
    if formula_id == "uniform_acceleration" and set(param_names) == {"a", "v0", "x0"}:
        c2, c1, c0 = np.polyfit(t_train, x_train, 2)
        seeds.insert(0, _clip_guess(np.asarray([2.0 * c2, c1, c0], dtype=float), lower, upper))
    return seeds


def fit_single_formula(
    formula: dict,
    t_train: np.ndarray,
    x_train: np.ndarray,
    t_test: np.ndarray,
    x_test: np.ndarray,
    formula_functions: Dict[str, FormulaFn] | None = None,
    maxfev: int = 20000,
) -> FitResult:
    formula_functions = formula_functions or FORMULA_FUNCTIONS
    formula_id = formula["id"]
    param_names = list(formula.get("params", []))
    num_params = len(param_names)
    fit_func = formula_functions.get(formula_id)

    if fit_func is None:
        return FitResult(
            formula_id=formula_id,
            success=False,
            params={},
            mse_train=float("inf"),
            mse_test=float("inf"),
            aic=float("inf"),
            aicc=float("inf"),
            score=float("inf"),
            passed=False,
            message=f"Unknown formula id '{formula_id}'",
        )

    lower, upper = _build_bounds(param_names, formula.get("bounds"))
    seeds = _initial_guesses(formula, t_train, x_train, lower, upper)

    best_popt = None
    best_train_mse = float("inf")
    last_error = "curve_fit failed for all seeds"

    for seed in seeds:
        try:
            popt, _ = curve_fit(
                fit_func,
                t_train,
                x_train,
                p0=seed,
                bounds=(lower, upper),
                maxfev=maxfev,
            )
        except Exception as exc:  # pragma: no cover - depends on numerical stability
            last_error = f"curve_fit failed: {exc}"
            continue

        train_mse = mse(x_train, fit_func(t_train, *popt))
        if train_mse < best_train_mse:
            best_train_mse = train_mse
            best_popt = popt

    if best_popt is None:
        return FitResult(
            formula_id=formula_id,
            success=False,
            params={},
            mse_train=float("inf"),
            mse_test=float("inf"),
            aic=float("inf"),
            aicc=float("inf"),
            score=float("inf"),
            passed=False,
            message=last_error,
        )

    y_train_pred = fit_func(t_train, *best_popt)
    y_test_pred = fit_func(t_test, *best_popt)
    mse_train = mse(x_train, y_train_pred)
    mse_test = mse(x_test, y_test_pred)
    aic = compute_aic(mse_train=mse_train, sample_size=t_train.size, num_params=num_params)
    aicc = compute_aicc(aic=aic, sample_size=t_train.size, num_params=num_params)
    params = {name: float(value) for name, value in zip(param_names, best_popt)}
    passed = bool(np.isfinite(mse_test) and mse_test <= TEST_MSE_THRESHOLD)

    return FitResult(
        formula_id=formula_id,
        success=True,
        params=params,
        mse_train=mse_train,
        mse_test=mse_test,
        aic=aic,
        aicc=aicc,
        score=float("inf"),
        passed=passed,
        message="" if passed else f"Test MSE {mse_test:.6g} exceeds threshold {TEST_MSE_THRESHOLD}",
    )


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    finite_values = [item for item in values if np.isfinite(item)]
    if not finite_values:
        return [0.0 for _ in values]
    min_v = min(finite_values)
    max_v = max(finite_values)
    if np.isclose(max_v - min_v, 0.0):
        return [0.0 for _ in values]
    return [
        0.0 if not np.isfinite(item) else (item - min_v) / (max_v - min_v)
        for item in values
    ]


def rank_results(
    results: Iterable[FitResult],
    aic_weight: float = AIC_WEIGHT,
) -> List[FitResult]:
    ok_results = [result for result in results if result.success]
    failed_results = [result for result in results if not result.success]

    if ok_results:
        test_norm = _normalize([result.mse_test for result in ok_results])
        info_norm = _normalize(
            [
                result.aicc if np.isfinite(result.aicc) else result.aic
                for result in ok_results
            ]
        )
        for idx, result in enumerate(ok_results):
            # Diagnostic score only. Decision ranking prioritizes extrapolation
            # MSE, then information criterion as a tie-breaker.
            result.score = float(test_norm[idx] + (aic_weight * info_norm[idx]))
        ok_results.sort(
            key=lambda item: (
                item.mse_test,
                item.aicc if np.isfinite(item.aicc) else item.aic,
                item.score,
                item.formula_id,
            )
        )

    failed_results.sort(key=lambda item: item.formula_id)
    return ok_results + failed_results


def fit_and_select(
    candidate_formulas: Sequence[dict],
    t_train: Sequence[float],
    x_train: Sequence[float],
    t_test: Sequence[float],
    x_test: Sequence[float],
    formula_functions: Dict[str, FormulaFn] | None = None,
    aic_weight: float = AIC_WEIGHT,
    test_mse_threshold: float = TEST_MSE_THRESHOLD,
    maxfev: int = 20000,
) -> dict:
    if not candidate_formulas:
        raise ValueError("candidate_formulas is empty")

    t_train_arr, x_train_arr = validate_series(
        t_train, x_train, require_strictly_increasing_time=True, min_points=3, name="train"
    )
    t_test_arr, x_test_arr = validate_series(
        t_test, x_test, require_strictly_increasing_time=True, min_points=1, name="test"
    )
    if t_train_arr[-1] >= t_test_arr[0]:
        raise ValueError("test times must start after the last train time for extrapolation")

    raw_results = [
        fit_single_formula(
            formula=formula,
            t_train=t_train_arr,
            x_train=x_train_arr,
            t_test=t_test_arr,
            x_test=x_test_arr,
            formula_functions=formula_functions,
            maxfev=maxfev,
        )
        for formula in candidate_formulas
    ]

    for result in raw_results:
        if result.success:
            result.passed = bool(
                np.isfinite(result.mse_test) and result.mse_test <= test_mse_threshold
            )
            if result.passed:
                result.message = ""
            else:
                result.message = (
                    f"Test MSE {result.mse_test:.6g} exceeds threshold {test_mse_threshold}"
                )

    ranked = rank_results(raw_results, aic_weight=aic_weight)
    best_candidate = next((item for item in ranked if item.success), None)
    accepted = next((item for item in ranked if item.success and item.passed), None)
    gate_failed = accepted is None

    payload = {
        # Preferred names:
        "best_candidate": best_candidate.to_dict() if best_candidate else None,
        "accepted": accepted.to_dict() if accepted else None,
        # Backward-compatible aliases:
        # - winner historically meant best ranked fit, even if gate failed.
        # - Use accepted/winner_passed for gate success.
        "winner": best_candidate.to_dict() if best_candidate else None,
        "winner_passed": accepted.to_dict() if accepted else None,
        "gate_failed": gate_failed,
        "should_fallback_to_discovery": gate_failed,
        "test_mse_threshold": float(test_mse_threshold),
        "ranked_results": [item.to_dict() for item in ranked],
    }
    return json_safe(payload)
