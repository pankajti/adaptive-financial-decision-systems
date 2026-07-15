"""Event selection and action-return construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig

ACTION_COLS = ["FOLLOW_TREND_return", "FOLLOW_MOMENTUM_return", "MEAN_REVERT_return", "ABSTAIN_return"]
ACTIONS = [c.replace("_return", "") for c in ACTION_COLS]


def add_event_and_action_returns(df: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    """Create event flags and tradable action-return columns.

    The action returns use next-open entry and close-at-horizon exit through the forward-return
    columns created in feature engineering.
    """
    df = df.copy()
    fwd_col = f"fwd_return_{cfg.HORIZON_DAYS}d"
    if fwd_col not in df.columns:
        raise ValueError(f"Missing {fwd_col}. Rebuild features with the requested horizon.")
    cost = cfg.COST_BPS / 10_000.0
    df["event_flag"] = (
        (df["dislocation_zscore_20"].abs() >= cfg.DISLOCATION_Z_THRESHOLD)
        | (df["momentum_score"].abs() >= cfg.MOMENTUM_Z_THRESHOLD)
        | (df["range_shock_20"] >= cfg.RANGE_SHOCK_THRESHOLD)
    )

    def action_return(direction_col: str) -> pd.Series:
        direction = df[direction_col].replace(0, np.nan)
        traded = direction.notna().astype(float)
        return (direction * df[fwd_col]).fillna(0.0) - traded * cost

    df["FOLLOW_TREND_return"] = action_return("trend_dir")
    df["FOLLOW_MOMENTUM_return"] = action_return("momentum_dir")
    df["MEAN_REVERT_return"] = action_return("mean_reversion_dir")
    df["ABSTAIN_return"] = 0.0
    df["oracle_best_action"] = df[ACTION_COLS].idxmax(axis=1).str.replace("_return", "", regex=False)
    df["oracle_best_return"] = df[ACTION_COLS].max(axis=1)
    if cfg.EVENT_ONLY:
        df = df[df["event_flag"]].copy()
    return df.dropna(subset=[fwd_col]).copy()


def prepare_model_frame(model_base: pd.DataFrame, cfg: ResearchConfig, feature_cols: list[str]) -> pd.DataFrame:
    """Create action returns and drop rows missing features/actions."""
    model_df = add_event_and_action_returns(model_base, cfg)
    return model_df.dropna(subset=feature_cols + ACTION_COLS).sort_values(["date", "symbol"]).reset_index(drop=True)
