#!/usr/bin/env python3
"""Newton CSV -> fit CLI (MVP orchestration).

Does NOT claim SAM2 / VLM / LLM / Genesis are implemented.
It only wires the existing perception + formula registry + fitting core.

Examples:
  python3 run_pipeline.py --csv data/samples/uniform_accel_meters.csv
  python3 run_pipeline.py --csv data/samples/uniform_accel_pixels.csv --pixels-per-meter 500
  python3 run_pipeline.py --csv path/to/traj.csv --candidates uniform_acceleration,uniform_linear_motion
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import config
from modules.fitting import fit_and_select, split_train_test_time_series
from modules.formula_registry import (
    build_candidate_formulas,
    flatten_formula_catalog,
    load_formula_catalog,
)
from modules.perception import convert_pixel_csv_to_meter_csv


def _read_meter_csv(path: Path) -> Tuple[List[float], List[float], List[float]]:
    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fields = set(reader.fieldnames)
        if "time" not in fields:
            raise ValueError("CSV must contain a 'time' column")
        if "x_meter" in fields:
            x_key = "x_meter"
            y_key = "y_meter" if "y_meter" in fields else None
        elif "x" in fields:
            x_key = "x"
            y_key = "y" if "y" in fields else None
        else:
            raise ValueError(
                "CSV must contain 'x_meter' (preferred) or 'x'. "
                "For pixel CSVs, pass --pixels-per-meter."
            )
        for row in reader:
            times.append(float(row["time"]))
            xs.append(float(row[x_key]))
            ys.append(float(row[y_key]) if y_key else 0.0)
    if len(times) < 6:
        raise ValueError(f"Need at least 6 rows, got {len(times)}")
    return times, xs, ys


def _resolve_input_csv(
    csv_path: Path,
    pixels_per_meter: float | None,
    converted_out: Path | None,
) -> Path:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])

    if "x_meter" in fields or ("x" in fields and "x_pix" not in fields):
        return csv_path

    if "x_pix" not in fields:
        raise ValueError(
            "Unrecognized CSV schema. Expected columns: "
            "time,x_meter[,y_meter] or time,x_pix,y_pix"
        )

    ppm = pixels_per_meter if pixels_per_meter is not None else config.PIXELS_PER_METER
    if ppm <= 0:
        raise ValueError(
            "Pixel CSV detected but pixels_per_meter is unset. "
            "Pass --pixels-per-meter or set config.PIXELS_PER_METER."
        )

    out_path = converted_out or csv_path.with_name(f"{csv_path.stem}_meters.csv")
    convert_pixel_csv_to_meter_csv(csv_path, out_path, pixels_per_meter=ppm)
    return out_path


def _default_candidate_ids(catalog: Sequence[dict]) -> List[str]:
    return [item["id"] for item in catalog]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Newton MVP: CSV trajectory -> formula fit + extrapolation gate"
    )
    parser.add_argument("--csv", required=True, help="Trajectory CSV path")
    parser.add_argument(
        "--axis",
        choices=("x", "y"),
        default="x",
        help="Which position axis to fit (default: x)",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="Required when input CSV uses x_pix/y_pix",
    )
    parser.add_argument(
        "--converted-csv",
        type=Path,
        default=None,
        help="Optional output path for pixel->meter conversion",
    )
    parser.add_argument(
        "--formulas",
        default="physics_formulas.json",
        help="Path to formula catalog JSON",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Comma-separated formula ids (default: all in catalog)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=config.SPLIT_RATIO,
        help="Train fraction for chronological split",
    )
    parser.add_argument(
        "--test-mse-threshold",
        type=float,
        default=config.TEST_MSE_THRESHOLD,
        help="Extrapolation MSE gate; fail => should_fallback_to_discovery",
    )
    parser.add_argument(
        "--aic-weight",
        type=float,
        default=config.AIC_WEIGHT,
        help="Weight of AICc in ranking score",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full result JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        return 2

    try:
        meter_csv = _resolve_input_csv(
            csv_path=csv_path,
            pixels_per_meter=args.pixels_per_meter,
            converted_out=args.converted_csv,
        )
        times, xs, ys = _read_meter_csv(meter_csv)
        positions = xs if args.axis == "x" else ys

        t_train, x_train, t_test, x_test = split_train_test_time_series(
            t_data=times,
            x_data=positions,
            split_ratio=args.split_ratio,
        )

        catalog = flatten_formula_catalog(load_formula_catalog(args.formulas))
        if args.candidates:
            candidate_ids = [item.strip() for item in args.candidates.split(",") if item.strip()]
        else:
            candidate_ids = _default_candidate_ids(catalog)

        candidates = build_candidate_formulas(catalog, candidate_ids)
        if not candidates:
            raise ValueError(f"No candidates resolved from: {candidate_ids}")

        result = fit_and_select(
            candidate_formulas=candidates,
            t_train=t_train,
            x_train=x_train,
            t_test=t_test,
            x_test=x_test,
            aic_weight=args.aic_weight,
            test_mse_threshold=args.test_mse_threshold,
        )
        result["meta"] = {
            "input_csv": str(csv_path),
            "meter_csv": str(meter_csv),
            "axis": args.axis,
            "n_points": len(times),
            "n_train": len(t_train),
            "n_test": len(t_test),
            "candidate_ids": candidate_ids,
            "not_implemented": [
                "SAM2 tracking",
                "VLM intuition",
                "LLM hypothesis generation",
                "Genesis imagination",
            ],
        }
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")

    if result.get("gate_failed"):
        print(
            "[gate] FAIL: no formula passed Test MSE threshold; "
            "should_fallback_to_discovery=true",
            file=sys.stderr,
        )
        return 3

    accepted = result.get("accepted") or {}
    print(
        f"[gate] PASS: accepted={accepted.get('formula_id')} "
        f"test_mse={accepted.get('mse_test')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
