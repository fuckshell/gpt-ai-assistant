"""Small executable demo for the fitting pipeline.

Run:
    python newton_fitting_demo.py
"""

from __future__ import annotations

import json

import numpy as np

import config
from modules.fitting import fit_and_select, split_train_test_time_series
from modules.formula_registry import (
    build_candidate_formulas,
    flatten_formula_catalog,
    load_formula_catalog,
)


def main() -> None:
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 4.0, 120)
    x_clean = 0.5 * 3.2 * t**2 + 1.0 * t + 0.2
    x = x_clean + rng.normal(0.0, 0.02, size=t.shape)

    t_train, x_train, t_test, x_test = split_train_test_time_series(
        t_data=t,
        x_data=x,
        split_ratio=config.SPLIT_RATIO,
    )

    catalog = flatten_formula_catalog(load_formula_catalog("physics_formulas.json"))
    candidates = build_candidate_formulas(
        formula_catalog=catalog,
        candidate_ids=[
            "uniform_linear_motion",
            "uniform_acceleration",
            "simple_harmonic_motion",
        ],
    )

    result = fit_and_select(
        candidate_formulas=candidates,
        t_train=t_train,
        x_train=x_train,
        t_test=t_test,
        x_test=x_test,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

