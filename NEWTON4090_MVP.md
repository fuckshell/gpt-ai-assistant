# Newton-4090 MVP (v3.1 rigorous baseline)

## Where this code lives

Newton is currently co-located inside the `gpt-ai-assistant` GitHub repo, on branch:

- `cursor/newton-cli-tests-6d37` (latest CLI/tests/gate work)
- earlier bootstrap also on `cursor/-bc-9211abd8-5300-4c43-93ce-e6da51916d37-e25e`

If your local workspace is an empty `.git` / empty `master` with no Newton files, you are **not** on the branch that contains this MVP. Sync with:

```bash
git fetch origin
git checkout cursor/newton-cli-tests-6d37
# or clone the remote and check out that branch
```

Do **not** rebuild from scratch just because a fresh empty local folder is missing files.

## What is included now

- `config.py`
  - `PIXELS_PER_METER = 0.0` (to be calibrated)
  - `SPLIT_RATIO = 0.8`
  - `TEST_MSE_THRESHOLD = 1e-3` (hard extrapolation gate)
  - `AIC_WEIGHT = 0.2`
- `physics_formulas.json`
  - Baseline kinematics and dynamics formula catalog
  - Parameter names and bounds aligned for safe fitting
- `modules/fitting.py`
  - train/test time-series split
  - `curve_fit` for each candidate model
  - metrics: `MSE`, `AIC`, `AICc`, `Test_MSE`
  - combined ranking score (`Test_MSE` + weighted information criterion)
  - hard gate: `passed` / `accepted` / `gate_failed` / `should_fallback_to_discovery`
- `modules/perception.py`
  - calibration helpers (`pixels_per_meter`)
  - pixel CSV -> meter CSV conversion
- `modules/discovery.py`
  - PySR wrapper
  - dimensional consistency checker
  - post-search penalty for dimensionally invalid equations
- `run_pipeline.py`
  - CSV -> fit CLI (highest-priority orchestration)
- `newton_fitting_demo.py`
  - synthetic data demo for immediate validation
- `data/samples/*.csv`
  - sample meter / pixel trajectories
- `tests_newton/test_newton_core.py`
  - unit tests for perception / fitting / CLI

## Exit codes (`run_pipeline.py`)

| Code | Meaning |
|---|---|
| 0 | Gate passed (`accepted` present) |
| 1 | Invalid input / expected runtime error |
| 2 | Missing CSV file |
| 3 | Gate failed (`should_fallback_to_discovery=true`) |

## Result field names

- `best_candidate`: best ranked successful fit (may fail gate)
- `accepted` / `winner_passed`: gate-passing formula (use this for success)
- `winner`: legacy alias of `best_candidate` (do **not** treat as gate success)


## Quick start

1. Install Python dependencies:

   ```bash
   python3 -m pip install -r requirements-newton.txt
   ```

2. Run fitting demo:

   ```bash
   python3 newton_fitting_demo.py
   ```

3. Run CSV pipeline CLI:

   ```bash
   python3 run_pipeline.py --csv data/samples/uniform_accel_meters.csv
   python3 run_pipeline.py --csv data/samples/uniform_accel_pixels.csv --pixels-per-meter 500
   ```

4. Run tests:

   ```bash
   python3 -m unittest tests_newton/test_newton_core.py -v
   ```

5. Calibrate perception stage:
   - set `PIXELS_PER_METER` in `config.py`, or
   - pass `--pixels-per-meter`, or
   - call `interactive_calibration()` in `modules/perception.py`

## Suggested release scope (MVP v1)

Only support:

- fixed camera
- known ruler/reference scale
- single-object 1D motion

Do not include multi-body interactions in v1.

## Recommended next milestones

1. Add `modules/reasoning.py` for Top-3 candidate generation from VLM/LLM hints.
2. Add Hybrid Tracker: motion spotter -> SAM2 -> trajectory CSV.
3. Wire discovery fallback when `should_fallback_to_discovery=true`.
4. Add bootstrap confidence intervals for fitted parameters.
5. Add rolling-origin validation in addition to the single 80/20 split.
6. Integrate Genesis replay as a soft verification layer.
7. Split Newton into its own repository (recommended; currently parasitic on LINE bot repo).
