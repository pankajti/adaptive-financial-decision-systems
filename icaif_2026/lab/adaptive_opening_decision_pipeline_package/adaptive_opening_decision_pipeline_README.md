# Adaptive Opening Decision Pipeline

This package implements a sector-free adaptive financial decision system for extreme opening moves.

## Files

- `adaptive_opening_decision_pipeline.ipynb` — runnable research notebook
- `adaptive_opening_decision_pipeline.py` — reusable pipeline module
- `adaptive_opening_decision_pipeline_README.md` — this file

## Action space

- `CONTINUE`: trade in the direction of the opening move
- `REVERSE`: trade against the opening move
- `ABSTAIN`: do not trade

## Experiments implemented

1. Opening-event labelling
2. Stock-level OHLC overreaction memory
3. Market-wide dispersion/context features
4. Latent behavioural regime discovery
5. Adaptive continue/reverse/abstain policy
6. Date-block Monte Carlo, cost sensitivity, and Parkinson-volatility barrier simulation

## Data requirements

Intraday data columns:

```text
symbol, timestamp, open, high, low, close, volume
```

Daily data columns:

```text
symbol, date, open, high, low, close, volume
```

The notebook runs on synthetic data by default. Set `USE_SYNTHETIC = False` in the data-loading cell to use your real data.

## Important design point

Sector/industry is intentionally excluded from the model input. If you have sector columns, keep them for post-hoc interpretation only.
