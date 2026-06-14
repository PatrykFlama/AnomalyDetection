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

This runs all NAB categories by default. Use `--categories` to select a subset.

Fast smoke test:

```powershell
uv run python scripts\run_experiment.py --categories realKnownCause --limit-series 2 --methods robust-zscore rolling-residual --ts2vec-iters 1 --ts2vec-batch-size 1 --ts2vec-max-train-length 64 --device cpu --ts2vec-dir ..\ts2vec
```

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
