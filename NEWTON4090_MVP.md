# Newton-4090 MVP (v3.1 rigorous baseline)

This repository now includes a minimal, publishable baseline for the Newton v3.1 pipeline.

## What is included now

- `config.py`
  - `PIXELS_PER_METER = 0.0` (to be calibrated)
  - `SPLIT_RATIO = 0.8`
- `physics_formulas.json`
  - Baseline kinematics and dynamics formula catalog
  - Parameter names and bounds aligned for safe fitting
- `modules/fitting.py`
  - train/test time-series split
  - `curve_fit` for each candidate model
  - metrics: `MSE`, `AIC`, `AICc`, `Test_MSE`
  - combined ranking score (`Test_MSE` + weighted information criterion)
- `modules/perception.py`
  - calibration helpers (`pixels_per_meter`)
  - pixel CSV -> meter CSV conversion
- `modules/discovery.py`
  - PySR wrapper
  - dimensional consistency checker
  - post-search penalty for dimensionally invalid equations
- `newton_fitting_demo.py`
  - synthetic data demo for immediate validation

## Quick start

1. Install Python dependencies:

   ```bash
   pip install numpy scipy pysr
   ```

2. Run fitting demo:

   ```bash
   python newton_fitting_demo.py
   ```

3. Calibrate perception stage:
   - set `PIXELS_PER_METER` in `config.py`, or
   - call `interactive_calibration()` in `modules/perception.py`

## Suggested release scope (MVP v1)

Only support:

- fixed camera
- known ruler/reference scale
- single-object 1D motion

Do not include multi-body interactions in v1.

## Recommended next milestones

1. Add `modules/reasoning.py` for Top-3 candidate generation from VLM/LLM hints.
2. Add a script to run full pipeline from `trajectory_data.csv`.
3. Add bootstrap confidence intervals for fitted parameters.
4. Add rolling-origin validation in addition to the single 80/20 split.
5. Integrate Genesis replay as a soft verification layer.

