# Agri ETF Adaptive Market-State Learning — Refactored Package

This refactor externalizes reusable notebook definitions into a Python package named `agri_market_state`.

## Files

- `agri_etf_adaptive_market_state_learning_refactored.ipynb` — clean notebook with imports only; no local `def` or `class` blocks.
- `agri_market_state/config.py` — `ResearchConfig` and `RANDOM_STATE`.
- `agri_market_state/data.py` — OHLCV normalization, yfinance loading, CSV loading, synthetic data.
- `agri_market_state/features.py` — trend, momentum, mean-reversion, volatility and cross-ETF context features.
- `agri_market_state/actions.py` — event flags and candidate action returns.
- `agri_market_state/projections.py` — PCA and fold-safe latent-state feature construction.
- `agri_market_state/policy.py` — walk-forward adaptive policy.
- `agri_market_state/evaluation.py` — daily returns, summaries and regime diagnostics.
- `agri_market_state/stress.py` — cost sensitivity and block-bootstrap Monte Carlo.
- `agri_market_state/vae.py` — optional VAE latent projection.
- `agri_market_state/baselines.py` — benchmark suite: fixed actions, rule baseline, symbol memory, PCA-regime memory, no-projection ML, random, oracle.

## Usage

Place the notebook and the `agri_market_state/` directory in the same project folder. Then run the notebook.

Start with:

```python
cfg = ResearchConfig(DATA_MODE="synthetic")
```

Then switch to real ETF data when ready:

```python
cfg = ResearchConfig(DATA_MODE="yfinance")
```

## Smoke test

The package was smoke-tested on synthetic data with the full feature, PCA, adaptive policy and baseline-report path.
