"""Adaptive policy learning for trend/momentum/mean-reversion/abstention actions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import RANDOM_STATE, ResearchConfig
from .projections import make_augmented_features


def walk_forward_adaptive_policy(
    df: pd.DataFrame,
    feature_cols: list[str],
    action_cols: list[str],
    cfg: ResearchConfig,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Walk-forward expected-return policy.

    For each fold, the model predicts expected return for each available action and selects
    the highest predicted edge, unless it is below cfg.MIN_EDGE_BPS, in which case it abstains.
    """
    df = df.sort_values(["date", "symbol"]).copy()
    dates = np.array(sorted(df["date"].drop_duplicates()))
    outputs = []
    min_edge = cfg.MIN_EDGE_BPS / 10_000.0
    start = cfg.TRAIN_DAYS
    while start < len(dates):
        train_dates = dates[max(0, start - cfg.TRAIN_DAYS):start]
        test_dates = dates[start:min(start + cfg.TEST_DAYS, len(dates))]
        train = df[df["date"].isin(train_dates)].copy()
        test = df[df["date"].isin(test_dates)].copy()
        if len(train) < 200 or len(test) == 0:
            start += cfg.STEP_DAYS
            continue

        Xtr, Xte, regimes, Zte = make_augmented_features(
            train[feature_cols], test[feature_cols], cfg.N_PCA_COMPONENTS, cfg.N_CLUSTERS, random_state
        )
        preds = {}
        for action_col in action_cols:
            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(Xtr, train[action_col].values)
            preds[action_col.replace("_return", "")] = model.predict(Xte)

        pred_df = pd.DataFrame(preds, index=test.index)
        best_action = pred_df.idxmax(axis=1)
        best_pred = pred_df.max(axis=1)
        chosen_action = best_action.where(best_pred >= min_edge, "ABSTAIN")

        out = test[["date", "symbol"] + action_cols].copy()
        for col in pred_df.columns:
            out[f"pred_{col}"] = pred_df[col]
        out["selected_action"] = chosen_action.values
        out["predicted_edge"] = best_pred.values
        out["walkforward_regime"] = regimes
        for i in range(Zte.shape[1]):
            out[f"fold_PC{i+1}"] = Zte[:, i]
        out["adaptive_return"] = [row.get(f"{row['selected_action']}_return", 0.0) for _, row in out.iterrows()]
        outputs.append(out)
        start += cfg.STEP_DAYS

    if not outputs:
        raise ValueError("No walk-forward output. Reduce TRAIN_DAYS or check sample size.")
    return pd.concat(outputs, ignore_index=True)
