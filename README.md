# Anomaly Detection in Time Series

Semi-supervised anomaly detection on NAB time series. Each method trains on the clean
normal prefix before the first labelled anomaly, selects the threshold quantile on
validation series, and reports final metrics on held-out test series.

## Setup

```powershell
uv sync
uv run python scripts\download_data.py
git clone https://github.com/zhihanyue/ts2vec.git ..\ts2vec
```

## Run Everything

```powershell
uv run python scripts\run_experiment.py --ts2vec-dir ..\ts2vec
```

This is the original end-to-end baseline/TS2Vec workflow. It runs all NAB
categories by default. Use `--categories` to select a subset.

Fast smoke test:

```powershell
uv run python scripts\run_experiment.py --categories realKnownCause --limit-series 2 --methods robust-zscore rolling-residual --ts2vec-iters 1 --ts2vec-batch-size 1 --ts2vec-max-train-length 64 --device cpu --ts2vec-dir ..\ts2vec
```

## Optuna Workflow

The current experiment workflow is `scripts/sweep_nab_optuna.py`. It keeps the
evaluation protocol fixed while Optuna searches only model-specific
hyperparameters.

Run a small sweep:

```console
uv run python scripts/sweep_nab_optuna.py --model robust_zscore --n-trials 3
```

Evaluate one exact parameter set on all loaded series:

```console
uv run python scripts/sweep_nab_optuna.py --model robust_zscore --mode final --params-file outputs/optuna/nab-robust_zscore/robust_zscore/trial_0000/params.json
```

Use `--categories`, `--limit-series`, and `--dry-run` for smoke tests. TS2Vec
runs additionally require `--ts2vec-dir`.

## Results

Main result table:

```text
results/<run_id>/summary.csv
```

Detailed outputs are saved in:

```text
results/<run_id>/baselines/
results/<run_id>/ts2vec/
```

Reported event/window metrics use NAB anomaly windows, but they are not the official NAB
score.

## Analyze Results

Analyze the latest complete baseline and TS2Vec run:

```console
uv run python scripts/analyze_results.py
```

Analyze a specific run:

```console
uv run python scripts/analyze_results.py --run-dir results/<run_id>
```

The script writes CSV statistics, a Markdown report, and PNG plots to
`results/<run_id>/analysis/`.

NAB visual audit:
https://colab.research.google.com/drive/1XaCqNpYcEjAytN1z7weSUp-F5b_zLJX1?usp=sharing

## Project Structure

```text
scripts/sweep_nab_optuna.py                 model scoring and run orchestration
src/anomaly_detection/optuna_sweep/config.py      CLI and fixed protocol configuration
src/anomaly_detection/optuna_sweep/protocol.py    validation split and train-prefix handling
src/anomaly_detection/optuna_sweep/evaluation.py  metrics and result export
src/anomaly_detection/optuna_sweep/utils.py       serialization and runtime utilities
presentation/final_presentation.pdf               final presentation
```
