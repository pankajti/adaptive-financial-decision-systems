"""
Baseline benchmark harness for the Agri ETF Adaptive Market-State Learning project.

Use this after your main notebook has created:
    model_df
    ACTION_COLS = ["FOLLOW_TREND_return", "FOLLOW_MOMENTUM_return", "MEAN_REVERT_return", "ABSTAIN_return"]
    adaptive_output  # optional, from your actual implementation

The baselines are deliberately simple, transparent, and fold-safe.
They are designed to answer: does PCA/VAE/adaptive policy add value over simpler rules?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    RandomForestRegressor = None
    SimpleImputer = None
    Pipeline = None
    StandardScaler = None
    PCA = None


DEFAULT_ACTION_COLS = [
    "FOLLOW_TREND_return",
    "FOLLOW_MOMENTUM_return",
    "MEAN_REVERT_return",
    "ABSTAIN_return",
]


@dataclass
class BaselineConfig:
    train_days: int = 504
    test_days: int = 63
    min_edge_bps: float = 25.0
    random_state: int = 42
    n_pca_components: int = 3
    n_pca_bins: int = 4
    rf_n_estimators: int = 200
    rf_min_samples_leaf: int = 25
    bootstrap_blocks: int = 24
    bootstrap_sims: int = 1000


# -----------------------------------------------------------------------------
# Generic performance utilities
# -----------------------------------------------------------------------------

def _as_action(action_col: str) -> str:
    return action_col.replace("_return", "")


def _as_return_col(action: str) -> str:
    return action if action.endswith("_return") else f"{action}_return"


def daily_portfolio_returns(df: pd.DataFrame, return_col: str, date_col: str = "date") -> pd.Series:
    """Equal-weight daily portfolio return across available symbols/events."""
    if len(df) == 0:
        return pd.Series(dtype=float, name=return_col)
    return df.groupby(date_col)[return_col].mean().sort_index().rename(return_col)


def summarize_daily_returns(daily: pd.Series, label: str) -> dict:
    daily = pd.Series(daily).dropna().sort_index()
    if len(daily) == 0:
        return {
            "strategy": label,
            "n_days": 0,
            "mean_daily_return": np.nan,
            "ann_return_approx": np.nan,
            "ann_vol_approx": np.nan,
            "sharpe_approx": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "positive_day_rate": np.nan,
        }
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    ann_return = daily.mean() * 252
    ann_vol = daily.std(ddof=1) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol and not np.isnan(ann_vol) and ann_vol > 0 else np.nan
    return {
        "strategy": label,
        "n_days": len(daily),
        "mean_daily_return": daily.mean(),
        "ann_return_approx": ann_return,
        "ann_vol_approx": ann_vol,
        "sharpe_approx": sharpe,
        "cumulative_return": equity.iloc[-1] - 1.0,
        "max_drawdown": drawdown.min(),
        "positive_day_rate": (daily > 0).mean(),
    }


def action_distribution(df: pd.DataFrame, action_col: str = "baseline_action") -> pd.DataFrame:
    if action_col not in df:
        return pd.DataFrame(columns=["action", "count", "share"])
    out = df[action_col].value_counts(dropna=False).rename_axis("action").reset_index(name="count")
    out["share"] = out["count"] / out["count"].sum()
    return out


def block_bootstrap_terminal_returns(
    daily: pd.Series,
    n_sims: int = 1000,
    block_size: int = 24,
    random_state: int = 42,
) -> dict:
    """Simple block bootstrap for terminal return robustness."""
    x = pd.Series(daily).dropna().values
    if len(x) == 0:
        return {"terminal_mean": np.nan, "terminal_median": np.nan, "terminal_p05": np.nan, "terminal_p95": np.nan, "prob_positive": np.nan}
    rng = np.random.default_rng(random_state)
    terminals = []
    n = len(x)
    starts = np.arange(max(1, n - block_size + 1))
    for _ in range(n_sims):
        path = []
        while len(path) < n:
            s = rng.choice(starts)
            path.extend(x[s : min(n, s + block_size)])
        path = np.asarray(path[:n])
        terminals.append(np.prod(1.0 + path) - 1.0)
    terminals = np.asarray(terminals)
    return {
        "terminal_mean": np.mean(terminals),
        "terminal_median": np.median(terminals),
        "terminal_p05": np.quantile(terminals, 0.05),
        "terminal_p95": np.quantile(terminals, 0.95),
        "prob_positive": np.mean(terminals > 0),
    }


# -----------------------------------------------------------------------------
# Baseline constructors
# -----------------------------------------------------------------------------

def fixed_action_baseline(df: pd.DataFrame, action: str, action_cols: Sequence[str] = DEFAULT_ACTION_COLS) -> pd.DataFrame:
    """Always choose one action: FOLLOW_TREND, FOLLOW_MOMENTUM, MEAN_REVERT, or ABSTAIN."""
    out = df.copy()
    ret_col = _as_return_col(action)
    if ret_col not in out.columns:
        raise ValueError(f"Missing return column: {ret_col}")
    out["baseline_action"] = _as_action(ret_col)
    out["baseline_return"] = out[ret_col]
    return out


def rule_based_market_state_baseline(df: pd.DataFrame, min_abs_signal: float = 0.0) -> pd.DataFrame:
    """
    Transparent baseline:
      - Strong dislocation -> MEAN_REVERT
      - Else strong momentum -> FOLLOW_MOMENTUM
      - Else trend -> FOLLOW_TREND
      - Else ABSTAIN

    This is intentionally simple and does not use PCA/VAE or ML.
    """
    out = df.copy()
    trend = out.get("trend_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    momentum = out.get("momentum_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    disloc = out.get("dislocation_zscore_20", pd.Series(0.0, index=out.index)).fillna(0.0)

    choices = np.full(len(out), "ABSTAIN", dtype=object)
    choices[np.abs(trend) > min_abs_signal] = "FOLLOW_TREND"
    choices[np.abs(momentum) > max(1.0, min_abs_signal)] = "FOLLOW_MOMENTUM"
    choices[np.abs(disloc) > max(1.5, min_abs_signal)] = "MEAN_REVERT"

    out["baseline_action"] = choices
    out["baseline_return"] = 0.0
    for action in ["FOLLOW_TREND", "FOLLOW_MOMENTUM", "MEAN_REVERT", "ABSTAIN"]:
        ret_col = _as_return_col(action)
        mask = out["baseline_action"].eq(action)
        if ret_col in out.columns:
            out.loc[mask, "baseline_return"] = out.loc[mask, ret_col]
    return out


def oracle_best_baseline(df: pd.DataFrame, action_cols: Sequence[str] = DEFAULT_ACTION_COLS) -> pd.DataFrame:
    """Diagnostic upper bound only. Uses future returns and must never be treated as tradable."""
    out = df.copy()
    missing = [c for c in action_cols if c not in out]
    if missing:
        raise ValueError(f"Missing action return columns: {missing}")
    out["baseline_action"] = out[list(action_cols)].idxmax(axis=1).str.replace("_return", "", regex=False)
    out["baseline_return"] = out[list(action_cols)].max(axis=1)
    return out


def random_action_baseline(
    df: pd.DataFrame,
    action_cols: Sequence[str] = DEFAULT_ACTION_COLS,
    random_state: int = 42,
    action_probs: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Random non-informational benchmark."""
    out = df.copy()
    rng = np.random.default_rng(random_state)
    actions = [_as_action(c) for c in action_cols]
    probs = action_probs if action_probs is not None else np.repeat(1 / len(actions), len(actions))
    out["baseline_action"] = rng.choice(actions, size=len(out), p=probs)
    out["baseline_return"] = 0.0
    for action in actions:
        ret_col = _as_return_col(action)
        mask = out["baseline_action"].eq(action)
        out.loc[mask, "baseline_return"] = out.loc[mask, ret_col]
    return out


# -----------------------------------------------------------------------------
# Fold-safe adaptive but simple baselines
# -----------------------------------------------------------------------------

def walk_forward_symbol_best_action_baseline(
    df: pd.DataFrame,
    action_cols: Sequence[str] = DEFAULT_ACTION_COLS,
    cfg: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """
    Fold-safe baseline: for each test period, choose the historically best action for each symbol
    using only the training window. Fallback to global best action if a symbol has too little history.
    """
    df = df.sort_values(["date", "symbol"]).copy()
    dates = np.array(sorted(df["date"].drop_duplicates()))
    outputs = []
    for start in range(cfg.train_days, len(dates), cfg.test_days):
        train_dates = dates[max(0, start - cfg.train_days):start]
        test_dates = dates[start:min(start + cfg.test_days, len(dates))]
        train = df[df["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)].copy()
        if len(test) == 0 or len(train) == 0:
            continue
        global_best = train[list(action_cols)].mean().idxmax()
        symbol_best = train.groupby("symbol")[list(action_cols)].mean().idxmax(axis=1).to_dict()
        chosen_cols = test["symbol"].map(symbol_best).fillna(global_best)
        test["baseline_action"] = chosen_cols.str.replace("_return", "", regex=False)
        test["baseline_return"] = [test.iloc[i][col] for i, col in enumerate(chosen_cols)]
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def walk_forward_pca_regime_best_action_baseline(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    action_cols: Sequence[str] = DEFAULT_ACTION_COLS,
    cfg: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """
    Fold-safe PCA-regime baseline.
    Fits PCA only on train, bins train/test PC1 using train quantiles, and chooses the historically
    best action in each PC1 regime. No predictive ML model is used.
    """
    if Pipeline is None:
        raise ImportError("scikit-learn is required for PCA-regime baseline")
    df = df.sort_values(["date", "symbol"]).copy()
    dates = np.array(sorted(df["date"].drop_duplicates()))
    outputs = []
    for start in range(cfg.train_days, len(dates), cfg.test_days):
        train_dates = dates[max(0, start - cfg.train_days):start]
        test_dates = dates[start:min(start + cfg.test_days, len(dates))]
        train = df[df["date"].isin(train_dates)].copy()
        test = df[df["date"].isin(test_dates)].copy()
        if len(train) == 0 or len(test) == 0:
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=cfg.n_pca_components, random_state=cfg.random_state)),
        ])
        z_train = pipe.fit_transform(train[list(feature_cols)])
        z_test = pipe.transform(test[list(feature_cols)])
        thresholds = np.quantile(z_train[:, 0], np.linspace(0, 1, cfg.n_pca_bins + 1)[1:-1])
        train_regime = np.digitize(z_train[:, 0], thresholds)
        test_regime = np.digitize(z_test[:, 0], thresholds)
        train["_regime"] = train_regime
        regime_best = train.groupby("_regime")[list(action_cols)].mean().idxmax(axis=1).to_dict()
        global_best = train[list(action_cols)].mean().idxmax()
        chosen_cols = pd.Series(test_regime).map(regime_best).fillna(global_best).values
        test["baseline_action"] = pd.Series(chosen_cols).str.replace("_return", "", regex=False).values
        test["baseline_return"] = [test.iloc[i][col] for i, col in enumerate(chosen_cols)]
        test["baseline_regime"] = test_regime
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def walk_forward_no_projection_ml_baseline(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    action_cols: Sequence[str] = DEFAULT_ACTION_COLS,
    cfg: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """
    Fold-safe no-PCA/no-VAE ML baseline.
    Trains one RandomForestRegressor per action on raw engineered features only and chooses the best
    predicted action when the predicted edge exceeds cfg.min_edge_bps, else ABSTAIN.
    """
    if RandomForestRegressor is None:
        raise ImportError("scikit-learn is required for no-projection ML baseline")
    df = df.sort_values(["date", "symbol"]).copy()
    dates = np.array(sorted(df["date"].drop_duplicates()))
    outputs = []
    min_edge = cfg.min_edge_bps / 10_000.0
    for start in range(cfg.train_days, len(dates), cfg.test_days):
        train_dates = dates[max(0, start - cfg.train_days):start]
        test_dates = dates[start:min(start + cfg.test_days, len(dates))]
        train = df[df["date"].isin(train_dates)].copy()
        test = df[df["date"].isin(test_dates)].copy()
        if len(train) == 0 or len(test) == 0:
            continue
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        Xtr = prep.fit_transform(train[list(feature_cols)])
        Xte = prep.transform(test[list(feature_cols)])
        preds = {}
        for col in action_cols:
            if col == "ABSTAIN_return":
                preds[col] = np.zeros(len(test))
                continue
            model = RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            model.fit(Xtr, train[col].values)
            preds[col] = model.predict(Xte)
        pred_df = pd.DataFrame(preds, index=test.index)
        best_col = pred_df.idxmax(axis=1)
        best_edge = pred_df.max(axis=1)
        chosen = best_col.where(best_edge >= min_edge, "ABSTAIN_return")
        test["baseline_action"] = chosen.str.replace("_return", "", regex=False).values
        test["baseline_predicted_edge"] = best_edge.values
        test["baseline_return"] = [test.loc[idx, col] for idx, col in zip(test.index, chosen)]
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


# -----------------------------------------------------------------------------
# Report builder
# -----------------------------------------------------------------------------

def evaluate_baseline(df: pd.DataFrame, label: str, return_col: str = "baseline_return") -> dict:
    return summarize_daily_returns(daily_portfolio_returns(df, return_col), label)


def build_baseline_report(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    action_cols: Sequence[str] = DEFAULT_ACTION_COLS,
    cfg: BaselineConfig = BaselineConfig(),
    adaptive_output: Optional[pd.DataFrame] = None,
    adaptive_return_col: str = "adaptive_return",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run a complete baseline suite and return a comparison table plus detailed outputs."""
    outputs: dict[str, pd.DataFrame] = {}
    summaries = []

    # Fixed-rule baselines
    for action in [_as_action(c) for c in action_cols]:
        out = fixed_action_baseline(model_df, action, action_cols)
        label = f"FIXED_{action}"
        outputs[label] = out
        summaries.append(evaluate_baseline(out, label))

    # Simple rule-based market-state baseline
    out = rule_based_market_state_baseline(model_df)
    outputs["RULE_MARKET_STATE"] = out
    summaries.append(evaluate_baseline(out, "RULE_MARKET_STATE"))

    # Fold-safe train-only baselines
    out = walk_forward_symbol_best_action_baseline(model_df, action_cols, cfg)
    outputs["WF_SYMBOL_BEST_ACTION"] = out
    summaries.append(evaluate_baseline(out, "WF_SYMBOL_BEST_ACTION"))

    out = walk_forward_pca_regime_best_action_baseline(model_df, feature_cols, action_cols, cfg)
    outputs["WF_PCA_REGIME_BEST_ACTION"] = out
    summaries.append(evaluate_baseline(out, "WF_PCA_REGIME_BEST_ACTION"))

    out = walk_forward_no_projection_ml_baseline(model_df, feature_cols, action_cols, cfg)
    outputs["WF_NO_PROJECTION_ML"] = out
    summaries.append(evaluate_baseline(out, "WF_NO_PROJECTION_ML"))

    # Diagnostic baselines
    out = random_action_baseline(model_df, action_cols, cfg.random_state)
    outputs["RANDOM_ACTION"] = out
    summaries.append(evaluate_baseline(out, "RANDOM_ACTION_DIAGNOSTIC"))

    out = oracle_best_baseline(model_df, action_cols)
    outputs["ORACLE_BEST"] = out
    summaries.append(evaluate_baseline(out, "ORACLE_BEST_DIAGNOSTIC"))

    if adaptive_output is not None and len(adaptive_output) > 0 and adaptive_return_col in adaptive_output:
        summaries.append(summarize_daily_returns(daily_portfolio_returns(adaptive_output, adaptive_return_col), "ACTUAL_ADAPTIVE_POLICY"))
        outputs["ACTUAL_ADAPTIVE_POLICY"] = adaptive_output.copy()

    comparison = pd.DataFrame(summaries)
    if "cumulative_return" in comparison:
        comparison = comparison.sort_values("cumulative_return", ascending=False).reset_index(drop=True)
    return comparison, outputs


def robustness_table(outputs: dict[str, pd.DataFrame], cfg: BaselineConfig = BaselineConfig()) -> pd.DataFrame:
    rows = []
    for label, out in outputs.items():
        ret_col = "adaptive_return" if "adaptive_return" in out.columns and label == "ACTUAL_ADAPTIVE_POLICY" else "baseline_return"
        if ret_col not in out.columns:
            continue
        daily = daily_portfolio_returns(out, ret_col)
        row = {"strategy": label}
        row.update(block_bootstrap_terminal_returns(daily, cfg.bootstrap_sims, cfg.bootstrap_blocks, cfg.random_state))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("prob_positive", ascending=False).reset_index(drop=True)
