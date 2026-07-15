"""Performance, action-distribution and regime-diagnostic utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_portfolio_returns(df: pd.DataFrame, return_col: str) -> pd.Series:
    """Equal-weight daily portfolio returns from row-level action returns."""
    return df.groupby("date")[return_col].mean().sort_index()


def summarize_daily_returns(daily: pd.Series, label: str) -> dict:
    """Compute simple daily-return summary statistics."""
    daily = daily.dropna().sort_index()
    if len(daily) == 0:
        return {"strategy": label, "n_days": 0}
    equity = (1 + daily).cumprod()
    dd = equity / equity.cummax() - 1
    ann_ret = daily.mean() * 252
    ann_vol = daily.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
    return {
        "strategy": label,
        "n_days": len(daily),
        "mean_daily_return": daily.mean(),
        "ann_return_approx": ann_ret,
        "ann_vol_approx": ann_vol,
        "sharpe_approx": sharpe,
        "cumulative_return": equity.iloc[-1] - 1,
        "max_drawdown": dd.min(),
        "positive_day_rate": (daily > 0).mean(),
    }


def fixed_action_summary(model_df: pd.DataFrame, action_cols: list[str]) -> pd.DataFrame:
    """Summaries for fixed-action baselines."""
    return pd.DataFrame([
        summarize_daily_returns(daily_portfolio_returns(model_df, col), col.replace("_return", ""))
        for col in action_cols
    ])


def action_distribution(df: pd.DataFrame, action_col: str = "selected_action") -> pd.DataFrame:
    """Action share table."""
    if action_col not in df.columns:
        return pd.DataFrame(columns=["action", "share"])
    return df[action_col].value_counts(normalize=True).rename("share").reset_index().rename(columns={"index": "action", action_col: "action"})


def regime_interpretation(output: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected actions and returns by walk-forward regime."""
    return output.groupby("walkforward_regime").agg(
        n=("adaptive_return", "size"),
        mean_adaptive_return=("adaptive_return", "mean"),
        mean_predicted_edge=("predicted_edge", "mean"),
        trend_share=("selected_action", lambda s: (s == "FOLLOW_TREND").mean()),
        momentum_share=("selected_action", lambda s: (s == "FOLLOW_MOMENTUM").mean()),
        mean_revert_share=("selected_action", lambda s: (s == "MEAN_REVERT").mean()),
        abstain_share=("selected_action", lambda s: (s == "ABSTAIN").mean()),
    ).sort_values("mean_adaptive_return", ascending=False)
