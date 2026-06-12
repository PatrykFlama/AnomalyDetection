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
uv run python scripts\run_experiment.py --categories realKnownCause --ts2vec-dir ..\ts2vec
```

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
