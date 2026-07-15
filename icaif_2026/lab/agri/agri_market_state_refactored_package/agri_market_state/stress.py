"""Robustness and Monte Carlo stress-testing utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .actions import ACTION_COLS, add_event_and_action_returns
from .config import ResearchConfig, RANDOM_STATE
from .evaluation import daily_portfolio_returns, summarize_daily_returns
from .policy import walk_forward_adaptive_policy


def run_cost_sensitivity(cfg: ResearchConfig, model_base: pd.DataFrame, feature_cols: list[str], costs=(0, 2, 5, 10, 20)) -> pd.DataFrame:
    """Rerun adaptive policy under a grid of transaction-cost assumptions."""
    rows = []
    for cost in costs:
        cfg2 = ResearchConfig(**{**cfg.__dict__, "COST_BPS": float(cost)})
        df2 = add_event_and_action_returns(model_base, cfg2)
        df2 = df2.dropna(subset=feature_cols + ACTION_COLS).sort_values(["date", "symbol"]).reset_index(drop=True)
        try:
            out = walk_forward_adaptive_policy(df2, feature_cols, ACTION_COLS, cfg2)
            summary = summarize_daily_returns(daily_portfolio_returns(out, "adaptive_return"), f"cost_{cost}_bps")
            summary["cost_bps"] = cost
            rows.append(summary)
        except Exception as exc:
            rows.append({"strategy": f"cost_{cost}_bps", "cost_bps": cost, "error": str(exc)})
    return pd.DataFrame(rows)


def block_bootstrap_paths(daily_returns: pd.Series, n_paths: int = 1000, block_size: int = 10, horizon_days: int | None = None, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate cumulative return paths using block bootstrap on daily returns."""
    rng = np.random.default_rng(seed)
    r = daily_returns.dropna().values
    if horizon_days is None:
        horizon_days = len(r)
    if len(r) < block_size + 1:
        raise ValueError("Not enough daily returns for block bootstrap")
    starts = np.arange(0, len(r) - block_size + 1)
    paths = np.zeros((horizon_days, n_paths))
    for p in range(n_paths):
        sampled = []
        while len(sampled) < horizon_days:
            s = rng.choice(starts)
            sampled.extend(r[s:s + block_size])
        sampled = np.array(sampled[:horizon_days])
        paths[:, p] = np.cumprod(1 + sampled) - 1
    return pd.DataFrame(paths)


def monte_carlo_summary(paths: pd.DataFrame) -> pd.Series:
    """Summary of terminal returns from a block-bootstrap path matrix."""
    terminal = paths.iloc[-1]
    return pd.Series({
        "terminal_mean": terminal.mean(),
        "terminal_median": terminal.median(),
        "terminal_p05": terminal.quantile(0.05),
        "terminal_p95": terminal.quantile(0.95),
        "prob_positive": (terminal > 0).mean(),
    })
