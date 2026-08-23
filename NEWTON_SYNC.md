# Newton Handoff Sync Note

## Critical discrepancy

A local Windows workspace may show:

- empty `.git`
- `master` with no commits
- no `NEWTON4090_MVP.md` / `physics_formulas.json` / `modules/*.py`

That does **not** mean Newton was never implemented.

## Canonical location (as of this branch)

Remote repository: `https://github.com/fuckshell/gpt-ai-assistant`  
Branch with Newton MVP + CLI/tests: `cursor/newton-cli-tests-6d37`

Key paths on that branch:

- `config.py`
- `physics_formulas.json`
- `modules/fitting.py`
- `modules/perception.py`
- `modules/discovery.py`
- `modules/formula_registry.py`
- `run_pipeline.py`
- `newton_fitting_demo.py`
- `data/samples/`
- `tests_newton/`
- `NEWTON4090_MVP.md`

## Recovery command for empty local folder

```bash
git clone https://github.com/fuckshell/gpt-ai-assistant.git
cd gpt-ai-assistant
git checkout cursor/newton-cli-tests-6d37
python3 -m pip install -r requirements-newton.txt
python3 -m unittest tests_newton/test_newton_core.py -v
python3 run_pipeline.py --csv data/samples/uniform_accel_meters.csv
```

## Policy while recovering

Restore and maintain the existing Newton **core interfaces**.  
Do not invent SAM2 / LLM / Genesis as completed work.
