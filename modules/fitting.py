"""Model fitting and extrapolation-based model selection for Newton v3.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

from config import AIC_WEIGHT, TEST_MSE_THRESHOLD
from modules.formula_registry import FORMULA_FUNCTIONS, FormulaFn

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
        return asdict(self)


def split_train_test_time_series(
    t_data: Sequence[float],
    x_data: Sequence[float],
    split_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    t = np.asarray(t_data, dtype=float)
    x = np.asarray(x_data, dtype=float)
    if t.shape != x.shape:
        raise ValueError("t_data and x_data must have identical shapes")
    if t.size < 6:
        raise ValueError("Need at least 6 points to support train/test fitting")
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


def _build_initial_guess(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
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
    p0 = _build_initial_guess(lower, upper)

    try:
        popt, _ = curve_fit(
            fit_func,
            t_train,
            x_train,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
    except Exception as exc:  # pragma: no cover - depends on numerical stability
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
            message=f"curve_fit failed: {exc}",
        )

    y_train_pred = fit_func(t_train, *popt)
    y_test_pred = fit_func(t_test, *popt)
    mse_train = mse(x_train, y_train_pred)
    mse_test = mse(x_test, y_test_pred)
    aic = compute_aic(mse_train=mse_train, sample_size=t_train.size, num_params=num_params)
    aicc = compute_aicc(aic=aic, sample_size=t_train.size, num_params=num_params)
    params = {name: float(value) for name, value in zip(param_names, popt)}
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
    min_v = min(values)
    max_v = max(values)
    if np.isclose(max_v - min_v, 0.0):
        return [0.0 for _ in values]
    return [(item - min_v) / (max_v - min_v) for item in values]


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
            result.score = float(test_norm[idx] + (aic_weight * info_norm[idx]))
        ok_results.sort(key=lambda item: (item.score, item.mse_test, item.aicc, item.aic))

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

    t_train_arr = np.asarray(t_train, dtype=float)
    x_train_arr = np.asarray(x_train, dtype=float)
    t_test_arr = np.asarray(t_test, dtype=float)
    x_test_arr = np.asarray(x_test, dtype=float)

    if t_train_arr.shape != x_train_arr.shape:
        raise ValueError("Train arrays must have identical shape")
    if t_test_arr.shape != x_test_arr.shape:
        raise ValueError("Test arrays must have identical shape")
    if t_train_arr.size < 3 or t_test_arr.size < 1:
        raise ValueError("Insufficient samples for fit/extrapolation")

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

    # Re-apply threshold so callers can override config at select-time.
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
    winner = next((item for item in ranked if item.success), None)
    accepted = next((item for item in ranked if item.success and item.passed), None)
    gate_failed = accepted is None

    return {
        "winner": winner.to_dict() if winner else None,
        "accepted": accepted.to_dict() if accepted else None,
        "gate_failed": gate_failed,
        "should_fallback_to_discovery": gate_failed,
        "test_mse_threshold": float(test_mse_threshold),
        "ranked_results": [item.to_dict() for item in ranked],
    }

