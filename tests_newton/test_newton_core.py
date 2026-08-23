"""Unit tests for Newton MVP core (fitting / perception / pipeline / discovery dims)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from modules.discovery import is_dimensionally_consistent
from modules.fitting import (
    compute_aic,
    compute_aicc,
    fit_and_select,
    split_train_test_time_series,
    validate_series,
)
from modules.formula_registry import (
    build_candidate_formulas,
    flatten_formula_catalog,
    load_formula_catalog,
    validate_formula_catalog,
)
from modules.json_util import json_safe
from modules.perception import (
    calibrate_pixels_per_meter,
    convert_pixel_csv_to_meter_csv,
    parse_length_to_meters,
)
import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class TestPerception(unittest.TestCase):
    def test_parse_length(self) -> None:
        self.assertAlmostEqual(parse_length_to_meters("0.3m"), 0.3)
        self.assertAlmostEqual(parse_length_to_meters("30cm"), 0.3)
        self.assertAlmostEqual(parse_length_to_meters("120mm"), 0.12)

    def test_calibrate_ppm(self) -> None:
        ppm = calibrate_pixels_per_meter((0.0, 0.0), (300.0, 0.0), 0.3)
        self.assertAlmostEqual(ppm, 1000.0)

    def test_pixel_csv_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pix.csv"
            dst = Path(tmp) / "m.csv"
            src.write_text(
                "time,x_pix,y_pix\n0.0,100.0,50.0\n0.1,200.0,50.0\n",
                encoding="utf-8",
            )
            convert_pixel_csv_to_meter_csv(src, dst, pixels_per_meter=100.0)
            lines = dst.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "time,x_meter,y_meter")
            self.assertTrue(lines[1].startswith("0.0,1.0,0.5"))


class TestFitting(unittest.TestCase):
    def test_aic_helpers(self) -> None:
        aic = compute_aic(mse_train=0.01, sample_size=100, num_params=3)
        self.assertTrue(np.isfinite(aic))
        aicc = compute_aicc(aic=aic, sample_size=100, num_params=3)
        self.assertGreater(aicc, aic)

    def test_split_ratio(self) -> None:
        t = np.linspace(0, 1, 10)
        x = t.copy()
        tr, xr, te, xe = split_train_test_time_series(t, x, 0.8)
        self.assertEqual(len(tr) + len(te), 10)
        self.assertEqual(len(tr), 8)

    def test_rejects_non_monotonic_time(self) -> None:
        t = np.asarray([0.0, 2.0, 1.0, 3.0, 4.0, 5.0])
        x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            split_train_test_time_series(t, x, 0.8)

    def test_rejects_nan(self) -> None:
        t = np.linspace(0, 1, 10)
        x = t.copy()
        x[3] = np.nan
        with self.assertRaises(ValueError):
            validate_series(t, x)

    def test_selects_uniform_acceleration(self) -> None:
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 4.0, 120)
        x = 0.5 * 3.2 * t**2 + 1.0 * t + 0.2 + rng.normal(0.0, 0.02, size=t.size)
        t_train, x_train, t_test, x_test = split_train_test_time_series(
            t, x, config.SPLIT_RATIO
        )
        catalog = flatten_formula_catalog(
            load_formula_catalog(ROOT / "physics_formulas.json")
        )
        validate_formula_catalog(catalog)
        candidates = build_candidate_formulas(
            catalog,
            [
                "uniform_linear_motion",
                "uniform_acceleration",
                "simple_harmonic_motion",
            ],
        )
        result = fit_and_select(
            candidates,
            t_train,
            x_train,
            t_test,
            x_test,
            test_mse_threshold=1.0,
        )
        self.assertIsNotNone(result["best_candidate"])
        self.assertEqual(result["best_candidate"]["formula_id"], "uniform_acceleration")
        self.assertFalse(result["gate_failed"])
        self.assertIsNotNone(result["accepted"])

    def test_recovers_simple_harmonic_motion(self) -> None:
        t = np.linspace(0.0, 8.0, 200)
        x = 1.2 * np.cos(2.0 * t + 0.3) + 0.5
        t_train, x_train, t_test, x_test = split_train_test_time_series(t, x, 0.8)
        catalog = flatten_formula_catalog(
            load_formula_catalog(ROOT / "physics_formulas.json")
        )
        candidates = build_candidate_formulas(catalog, ["simple_harmonic_motion"])
        result = fit_and_select(
            candidates,
            t_train,
            x_train,
            t_test,
            x_test,
            test_mse_threshold=1e-4,
        )
        self.assertFalse(result["gate_failed"])
        params = result["accepted"]["params"]
        self.assertAlmostEqual(abs(params["A"]), 1.2, places=2)
        self.assertAlmostEqual(params["omega"], 2.0, places=2)

    def test_gate_failure_sets_fallback_flag(self) -> None:
        t = np.linspace(0.0, 2.0, 40)
        x = np.sin(20 * t) + np.cos(13 * t)
        t_train, x_train, t_test, x_test = split_train_test_time_series(t, x, 0.8)
        catalog = flatten_formula_catalog(
            load_formula_catalog(ROOT / "physics_formulas.json")
        )
        candidates = build_candidate_formulas(
            catalog, ["uniform_linear_motion", "uniform_acceleration"]
        )
        result = fit_and_select(
            candidates,
            t_train,
            x_train,
            t_test,
            x_test,
            test_mse_threshold=1e-12,
        )
        self.assertTrue(result["gate_failed"])
        self.assertTrue(result["should_fallback_to_discovery"])
        self.assertIsNone(result["accepted"])
        # Strict JSON: no Infinity tokens.
        text = json.dumps(result, allow_nan=False)
        self.assertNotIn("Infinity", text)

    def test_unknown_candidate_id_errors(self) -> None:
        catalog = flatten_formula_catalog(
            load_formula_catalog(ROOT / "physics_formulas.json")
        )
        with self.assertRaises(ValueError):
            build_candidate_formulas(
                catalog, ["uniform_acceleration", "typo"], strict_unknown=True
            )


class TestDiscoveryDims(unittest.TestCase):
    def test_quadratic_in_time_is_valid_with_free_constants(self) -> None:
        self.assertTrue(is_dimensionally_consistent("1.0 + 2.0 * x0 + 3.0 * x0**2"))

    def test_raw_time_alone_is_invalid_for_position_target(self) -> None:
        self.assertFalse(is_dimensionally_consistent("x0"))


class TestPipelineCLI(unittest.TestCase):
    def test_cli_on_sample_meters_csv(self) -> None:
        sample = ROOT / "data" / "samples" / "uniform_accel_meters.csv"
        self.assertTrue(sample.exists(), "sample CSV missing")
        code = run_pipeline.main(
            [
                "--csv",
                str(sample),
                "--candidates",
                "uniform_linear_motion,uniform_acceleration,simple_harmonic_motion",
                "--test-mse-threshold",
                "0.01",
            ]
        )
        self.assertEqual(code, 0)

    def test_cli_works_from_other_cwd(self) -> None:
        sample = ROOT / "data" / "samples" / "uniform_accel_meters.csv"
        with tempfile.TemporaryDirectory() as tmp:
            old = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                code = run_pipeline.main(
                    [
                        "--csv",
                        str(sample),
                        "--test-mse-threshold",
                        "0.01",
                    ]
                )
            finally:
                os.chdir(old)
        self.assertEqual(code, 0)

    def test_cli_missing_file_exit_2(self) -> None:
        code = run_pipeline.main(["--csv", str(ROOT / "missing.csv")])
        self.assertEqual(code, 2)

    def test_cli_axis_y_without_y_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "only_x.csv"
            path.write_text(
                "time,x_meter\n"
                + "\n".join(f"{i*0.1:.1f},{i}" for i in range(10))
                + "\n",
                encoding="utf-8",
            )
            code = run_pipeline.main(["--csv", str(path), "--axis", "y"])
        self.assertEqual(code, 1)

    def test_json_safe_helper(self) -> None:
        payload = json_safe({"a": float("inf"), "b": [float("nan"), 1.0]})
        text = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(text)["a"], None)


if __name__ == "__main__":
    unittest.main()
