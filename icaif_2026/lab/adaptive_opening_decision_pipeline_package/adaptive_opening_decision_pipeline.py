"""
Adaptive Opening Decision Pipeline
==================================

Sector-free adaptive financial decision system for extreme opening moves.

Core action space:
    CONTINUE: trade in the direction of the first-30-minute move
    REVERSE:  trade against the first-30-minute move
    ABSTAIN:  do not trade

This module intentionally excludes sector/industry as model input. Sector can be
used later only for post-hoc interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PipelineConfig:
    opening_start: str = "09:30"
    decision_time: str = "10:00"
    market_close: str = "16:00"
    extreme_quantile: float = 0.80
    overreaction_windows: Tuple[int, ...] = (60, 120)
    event_memory_windows: Tuple[int, ...] = (20, 60)
    min_periods_fraction: float = 0.5
    n_latent_regimes: int = 5
    cost_bps: float = 5.0
    min_edge_bps: float = 2.0
    random_state: int = 42


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.tz_localize(None).dt.normalize()


def _normalize_timestamp_series(
    s: pd.Series,
    source_tz: Optional[str] = None,
    market_tz: Optional[str] = None,
) -> pd.Series:
    """Return timezone-naive timestamps in market time.

    If `source_tz` is supplied for naive timestamps, timestamps are localized to
    that timezone first. If `market_tz` is supplied, timestamps are converted to
    that timezone and then made timezone-naive. This is useful when intraday bars
    are stored in UTC but opening/decision times are expressed in New York time.
    """
    ts = pd.to_datetime(s)

    # pandas exposes a single timezone for Series with datetimetz dtype.
    if getattr(ts.dt, "tz", None) is None:
        if source_tz is not None:
            ts = ts.dt.tz_localize(source_tz, nonexistent="shift_forward", ambiguous="NaT")
    elif source_tz is not None:
        # Already timezone-aware; keep original timezone instead of relocalizing.
        pass

    if market_tz is not None:
        if getattr(ts.dt, "tz", None) is None:
            # Assume timestamps are already in market-local wall-clock time.
            return ts.dt.tz_localize(None)
        return ts.dt.tz_convert(market_tz).dt.tz_localize(None)

    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_localize(None)
    return ts


def normalize_intraday_timestamps(
    intraday: pd.DataFrame,
    timestamp_col: str = "timestamp",
    source_tz: Optional[str] = "UTC",
    market_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Convert intraday timestamps to timezone-naive market-local timestamps.

    Example
    -------
    If bars are stored in UTC, 2025-09-05 13:30 becomes 2025-09-05 09:30
    in America/New_York during daylight saving time. Run this before applying
    opening-time rules such as 09:30 to 10:00.
    """
    _require_columns(intraday, [timestamp_col], "intraday")
    out = intraday.copy()
    out[timestamp_col] = _normalize_timestamp_series(out[timestamp_col], source_tz=source_tz, market_tz=market_tz)
    return out


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    eq = pd.Series(equity).astype(float)
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Synthetic data for smoke tests and demonstrations
# ---------------------------------------------------------------------------

def make_synthetic_intraday_data(
    n_symbols: int = 80,
    n_days: int = 90,
    bars_per_day: int = 13,
    start_date: str = "2026-01-02",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic 30-minute intraday OHLCV and derived daily OHLCV.

    This is only for smoke testing the pipeline when real market data is not
    mounted. It intentionally creates mild opening reversal behaviour in a subset
    of symbols so the pipeline has something to learn.
    """
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    dates = pd.bdate_range(start=start_date, periods=n_days)
    times = pd.date_range("09:30", periods=bars_per_day, freq="30min").time

    rows = []
    base_prices = rng.lognormal(mean=np.log(80), sigma=0.45, size=n_symbols)
    stock_reversal_tendency = rng.normal(0.0, 0.35, size=n_symbols)

    for d in dates:
        market_shock = rng.normal(0, 0.004)
        market_disp = abs(rng.normal(0.0, 0.003))
        for s_idx, symbol in enumerate(symbols):
            price = base_prices[s_idx] * np.exp(rng.normal(0, 0.002))
            stock_noise = rng.normal(0, 0.006 + market_disp)
            opening_ret = market_shock + stock_noise
            # Some stocks are more mean-reverting after extreme openings.
            reversal_alpha = stock_reversal_tendency[s_idx]
            post_drift = -reversal_alpha * opening_ret + rng.normal(0, 0.006)
            intraday_rets = rng.normal(post_drift / (bars_per_day - 1), 0.0025 + market_disp, bars_per_day)
            intraday_rets[0] = opening_ret

            current_open = price
            for b, tm in enumerate(times):
                timestamp = pd.Timestamp.combine(d.date(), tm)
                ret = intraday_rets[b]
                close = current_open * np.exp(ret)
                high = max(current_open, close) * (1 + abs(rng.normal(0, 0.0015)))
                low = min(current_open, close) * (1 - abs(rng.normal(0, 0.0015)))
                volume = int(rng.lognormal(12.0, 0.6))
                rows.append(
                    {
                        "timestamp": timestamp,
                        "date": pd.Timestamp(d).normalize(),
                        "symbol": symbol,
                        "open": current_open,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
                current_open = close
            base_prices[s_idx] = current_open

    intraday = pd.DataFrame(rows)
    daily = (
        intraday.sort_values(["symbol", "timestamp"])
        .groupby(["symbol", "date"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    return intraday, daily


# ---------------------------------------------------------------------------
# Experiment 1: Opening-event labelling
# ---------------------------------------------------------------------------

def compute_opening_events(
    intraday: pd.DataFrame,
    opening_start: str = "09:30",
    decision_time: str = "10:00",
    market_close: str = "16:00",
    extreme_quantile: float = 0.80,
    filter_extreme: bool = True,
    source_tz: Optional[str] = None,
    market_tz: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create opening-event labels from intraday bars.

    Parameters
    ----------
    intraday:
        Intraday OHLCV bars. Required columns are timestamp, symbol, open, close.
        high/low/volume are optional but recommended because they create valid
        first-30-minute decision-time features.
    source_tz, market_tz:
        Optional timezone handling. If bars are stored in UTC, pass
        source_tz="UTC", market_tz="America/New_York". The opening_start,
        decision_time, and market_close arguments are then interpreted in the
        market timezone. If timestamps are already market-local, leave both as
        None.

    Returns
    -------
    events:
        Extreme events only if filter_extreme=True; otherwise all stock-days.
    all_stock_days:
        All stock-days with opening/post-opening returns. Used for market context.
    """
    _require_columns(intraday, ["timestamp", "symbol", "open", "close"], "intraday")
    df = intraday.copy()
    df["timestamp"] = _normalize_timestamp_series(df["timestamp"], source_tz=source_tz, market_tz=market_tz)
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.normalize()
    else:
        # Recompute from converted timestamp if timezone conversion was requested;
        # otherwise trust the provided date after normalization.
        if source_tz is not None or market_tz is not None:
            df["date"] = df["timestamp"].dt.normalize()
        else:
            df["date"] = _to_date(df["date"])
    df["time"] = df["timestamp"].dt.strftime("%H:%M")
    df = df.sort_values(["symbol", "date", "timestamp"])

    has_high_low = {"high", "low"}.issubset(df.columns)
    has_volume = "volume" in df.columns
    rows: list[dict[str, Any]] = []

    for (symbol, date), row_group in df.groupby(["symbol", "date"], sort=False):
        after_open = row_group[row_group["time"] >= opening_start]
        decision_window = row_group[(row_group["time"] >= opening_start) & (row_group["time"] <= decision_time)]
        close_window = row_group[row_group["time"] <= market_close]
        if after_open.empty or decision_window.empty or close_window.empty:
            continue

        open_price = float(after_open.iloc[0]["open"])
        decision_price = float(decision_window.iloc[-1]["close"])
        close_price = float(close_window.iloc[-1]["close"])

        opening_volume = float(decision_window["volume"].sum()) if has_volume else np.nan
        total_volume = float(close_window["volume"].sum()) if has_volume else np.nan

        row: dict[str, Any] = {
            "symbol": symbol,
            "date": pd.Timestamp(date).normalize(),
            "open_price": open_price,
            "decision_price": decision_price,
            "close_price": close_price,
            "opening_volume": opening_volume,
            "total_volume": total_volume,
        }

        if has_high_low:
            opening_high = float(decision_window["high"].max())
            opening_low = float(decision_window["low"].min())
            opening_range = opening_high - opening_low
            row.update(
                {
                    "opening_high": opening_high,
                    "opening_low": opening_low,
                    "opening_range_pct": opening_range / open_price if open_price else np.nan,
                    "opening_close_location": (decision_price - opening_low) / opening_range if opening_range else np.nan,
                }
            )
        rows.append(row)

    expected_cols = [
        "symbol", "date", "open_price", "decision_price", "close_price",
        "opening_volume", "total_volume", "opening_high", "opening_low",
        "opening_range_pct", "opening_close_location",
    ]
    all_days = pd.DataFrame(rows)
    if all_days.empty:
        empty = pd.DataFrame(columns=expected_cols + [
            "opening_return", "post_opening_return", "abs_opening_return",
            "opening_direction", "opening_strength_rank", "is_extreme_opening",
            "continue_return", "reverse_return", "continuation_flag", "reversal_flag",
            "opening_directional_efficiency",
        ])
        return empty.copy(), empty.copy()

    for c in expected_cols:
        if c not in all_days.columns:
            all_days[c] = np.nan

    all_days["opening_return"] = all_days["decision_price"] / all_days["open_price"] - 1.0
    all_days["post_opening_return"] = all_days["close_price"] / all_days["decision_price"] - 1.0
    all_days["abs_opening_return"] = all_days["opening_return"].abs()
    all_days["opening_direction"] = np.sign(all_days["opening_return"]).astype(int)
    all_days = all_days[all_days["opening_direction"] != 0].copy()

    if "opening_range_pct" in all_days.columns:
        all_days["opening_directional_efficiency"] = _safe_divide(
            all_days["abs_opening_return"], all_days["opening_range_pct"]
        )

    all_days["opening_strength_rank"] = all_days.groupby("date")["abs_opening_return"].rank(pct=True, method="average")
    all_days["is_extreme_opening"] = all_days["opening_strength_rank"] >= extreme_quantile

    all_days["continue_return"] = all_days["opening_direction"] * all_days["post_opening_return"]
    all_days["reverse_return"] = -all_days["continue_return"]
    all_days["continuation_flag"] = (all_days["continue_return"] > 0).astype(int)
    all_days["reversal_flag"] = (all_days["reverse_return"] > 0).astype(int)

    events = all_days[all_days["is_extreme_opening"]].copy() if filter_extreme else all_days.copy()
    return events.reset_index(drop=True), all_days.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Experiment 2: Overreaction memory and range-based volatility
# ---------------------------------------------------------------------------

def add_daily_candle_features(daily: pd.DataFrame) -> pd.DataFrame:
    _require_columns(daily, ["symbol", "date", "open", "high", "low", "close"], "daily")
    df = daily.copy()
    df["date"] = _to_date(df["date"])
    df = df.sort_values(["symbol", "date"])

    df["body_return"] = df["close"] / df["open"] - 1.0
    df["range_pct"] = (df["high"] - df["low"]) / df["open"]
    df["open_to_high_pct"] = (df["high"] - df["open"]) / df["open"]
    df["open_to_low_pct"] = (df["open"] - df["low"]) / df["open"]
    df["close_location"] = _safe_divide(df["close"] - df["low"], df["high"] - df["low"])
    df["directional_efficiency"] = _safe_divide(df["body_return"].abs(), df["range_pct"])
    df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
    df["log_hl_sq"] = np.log(df["high"] / df["low"]).pow(2)
    return df


def add_overreaction_memory(
    daily: pd.DataFrame,
    windows: Sequence[int] = (60, 120),
    min_periods_fraction: float = 0.5,
    annualize_parkinson: bool = True,
) -> pd.DataFrame:
    """Compute rolling Becker-style upside/downside overreaction ratios.

    All rolling features are shifted by one day within each symbol to avoid
    using today's high/low/close at the opening-decision time.
    """
    df = add_daily_candle_features(daily)

    log_o = np.log(df["open"])
    log_h = np.log(df["high"])
    log_l = np.log(df["low"])
    log_c = np.log(df["close"])

    df["oc_log_return"] = log_c - log_o
    df["V_up"] = 2.0 * (log_h - log_o) * (log_h - log_c)
    df["V_down"] = 2.0 * (log_o - log_l) * (log_c - log_l)
    df["V_up"] = df["V_up"].clip(lower=0)
    df["V_down"] = df["V_down"].clip(lower=0)

    g = df.groupby("symbol", group_keys=False)
    for w in windows:
        minp = max(5, int(w * min_periods_fraction))
        shifted_v_up = g["V_up"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())
        shifted_v_down = g["V_down"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())
        shifted_var = g["oc_log_return"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).var(ddof=1))

        df[f"F_up_{w}"] = _safe_divide(shifted_v_up, shifted_var)
        df[f"F_down_{w}"] = _safe_divide(shifted_v_down, shifted_var)
        df[f"F_asymmetry_{w}"] = df[f"F_down_{w}"] - df[f"F_up_{w}"]

        parkinson_var = g["log_hl_sq"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean() / (4.0 * np.log(2.0)))
        if annualize_parkinson:
            df[f"parkinson_vol_{w}"] = np.sqrt(parkinson_var * 252.0)
        else:
            df[f"parkinson_vol_{w}"] = np.sqrt(parkinson_var)

        prior_range = g["range_pct"].transform(lambda s: s.shift(1))
        prior_range_median = g["range_pct"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).median())
        df[f"prior_range_shock_{w}"] = _safe_divide(prior_range, prior_range_median)

        # Additional fully lagged candle-memory features that are known at the
        # 10:00 decision time. These replace any current-day full-range feature.
        df[f"prior_range_pct_{w}"] = prior_range
        df[f"prior_close_location_{w}"] = g["close_location"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())
        df[f"prior_directional_efficiency_{w}"] = g["directional_efficiency"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())

    return df


def add_rolling_event_memory(
    events: pd.DataFrame,
    windows: Sequence[int] = (20, 60),
    min_periods_fraction: float = 0.5,
) -> pd.DataFrame:
    _require_columns(events, ["symbol", "date", "continue_return", "reverse_return", "reversal_flag"], "events")
    df = events.copy().sort_values(["symbol", "date"])
    g = df.groupby("symbol", group_keys=False)

    for w in windows:
        minp = max(3, int(w * min_periods_fraction))
        df[f"rolling_reversal_rate_{w}"] = g["reversal_flag"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())
        df[f"rolling_continue_return_{w}"] = g["continue_return"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())
        df[f"rolling_reverse_return_{w}"] = g["reverse_return"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())

        if "opening_volume" in df.columns:
            log_opening_volume = np.log1p(df["opening_volume"].astype(float))
            shifted = log_opening_volume.groupby(df["symbol"]).shift(1)
            roll_mean = shifted.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=minp).mean())
            roll_std = shifted.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=minp).std(ddof=1))
            df[f"opening_volume_zscore_{w}"] = _safe_divide(log_opening_volume - roll_mean, roll_std)

        if "opening_range_pct" in df.columns:
            shifted_range = df.groupby("symbol")["opening_range_pct"].shift(1)
            roll_range_median = shifted_range.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=minp).median())
            df[f"opening_range_shock_{w}"] = _safe_divide(df["opening_range_pct"], roll_range_median)

    return df


# ---------------------------------------------------------------------------
# Experiment 3: Market context and dispersion, no sector input
# ---------------------------------------------------------------------------

def compute_opening_market_context(all_stock_days: pd.DataFrame) -> pd.DataFrame:
    """Market context observable at decision time from first-30-minute moves."""
    _require_columns(all_stock_days, ["date", "opening_return", "abs_opening_return", "opening_direction"], "all_stock_days")
    ctx = (
        all_stock_days.groupby("date")
        .agg(
            market_opening_dispersion=("opening_return", "std"),
            market_opening_abs_median=("abs_opening_return", "median"),
            market_opening_abs_p90=("abs_opening_return", lambda x: x.quantile(0.90)),
            market_opening_breadth=("opening_return", lambda x: (x > 0).mean()),
            market_opening_mean=("opening_return", "mean"),
            active_symbols=("symbol", "nunique"),
        )
        .reset_index()
    )
    return ctx


def compute_lagged_daily_market_context(daily: pd.DataFrame) -> pd.DataFrame:
    """Prior-day market context from daily candles. Shifted by one date to avoid leakage."""
    df = add_daily_candle_features(daily)
    daily_ctx = (
        df.groupby("date")
        .agg(
            prior_market_body_dispersion=("body_return", "std"),
            prior_market_abs_return_median=("body_return", lambda x: x.abs().median()),
            prior_market_range_median=("range_pct", "median"),
            prior_market_range_p90=("range_pct", lambda x: x.quantile(0.90)),
            prior_market_breadth=("body_return", lambda x: (x > 0).mean()),
            prior_market_close_location_median=("close_location", "median"),
            prior_market_directional_efficiency=("directional_efficiency", "median"),
        )
        .sort_index()
    )
    daily_ctx = daily_ctx.shift(1).reset_index()
    return daily_ctx


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def assemble_model_frame(
    events: pd.DataFrame,
    all_stock_days: pd.DataFrame,
    daily: pd.DataFrame,
    config: PipelineConfig = PipelineConfig(),
) -> pd.DataFrame:
    """Merge event labels, stock memory, opening market context, and lagged daily context."""
    events_mem = add_rolling_event_memory(events, config.event_memory_windows, config.min_periods_fraction)
    daily_mem = add_overreaction_memory(daily, config.overreaction_windows, config.min_periods_fraction)
    opening_ctx = compute_opening_market_context(all_stock_days)
    lagged_daily_ctx = compute_lagged_daily_market_context(daily)

    # Keep only non-leaky rolling stock memory columns from daily_mem.
    memory_cols = [
        c for c in daily_mem.columns
        if (
            c.startswith("F_")
            or c.startswith("parkinson_vol_")
            or c.startswith("prior_range_shock_")
            or c.startswith("prior_range_pct_")
            or c.startswith("prior_close_location_")
            or c.startswith("prior_directional_efficiency_")
        )
    ]
    memory_cols += ["symbol", "date"]
    model_df = events_mem.merge(daily_mem[memory_cols], on=["symbol", "date"], how="left")
    model_df = model_df.merge(opening_ctx, on="date", how="left")
    model_df = model_df.merge(lagged_daily_ctx, on="date", how="left")
    return model_df.sort_values(["date", "symbol"]).reset_index(drop=True)


def default_feature_columns(model_df: pd.DataFrame) -> list[str]:
    forbidden = {
        "date", "symbol", "open_price", "decision_price", "close_price",
        "post_opening_return", "continue_return", "reverse_return",
        "continuation_flag", "reversal_flag", "is_extreme_opening",
        "total_volume", "close", "high", "low", "open"
    }
    prefixes = (
        "opening_", "abs_opening_return", "F_", "rolling_", "market_",
        "prior_market_", "parkinson_vol_", "prior_range_shock_",
        "prior_range_pct_", "prior_close_location_", "prior_directional_efficiency_"
    )
    cols = []
    for c in model_df.columns:
        if c in forbidden:
            continue
        if c == "opening_direction" or c.startswith(prefixes):
            if pd.api.types.is_numeric_dtype(model_df[c]):
                cols.append(c)
    # Remove target-like roll columns? rolling_continue/reverse returns are lagged, allowed.
    return sorted(set(cols))


# ---------------------------------------------------------------------------
# Experiment 4: Latent behavioural regimes
# ---------------------------------------------------------------------------

def fit_latent_regime_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    n_regimes: int = 5,
    method: str = "gmm",
    random_state: int = 42,
) -> Pipeline:
    """Fit unsupervised latent regime model on train data only."""
    if method not in {"gmm", "kmeans"}:
        raise ValueError("method must be 'gmm' or 'kmeans'")
    cluster: BaseEstimator
    if method == "gmm":
        cluster = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=random_state)
    else:
        cluster = KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("cluster", cluster),
        ]
    )
    pipe.fit(train_df[list(feature_cols)])
    return pipe


def add_latent_regime_features(
    df: pd.DataFrame,
    regime_model: Pipeline,
    feature_cols: Sequence[str],
    n_regimes: int,
) -> pd.DataFrame:
    out = df.copy()
    x = out[list(feature_cols)]
    cluster = regime_model.named_steps["cluster"]
    transformed = regime_model[:-1].transform(x)
    if hasattr(cluster, "predict_proba"):
        probs = cluster.predict_proba(transformed)
        out["latent_regime"] = probs.argmax(axis=1)
        for k in range(n_regimes):
            out[f"regime_prob_{k}"] = probs[:, k]
    else:
        labels = cluster.predict(transformed)
        out["latent_regime"] = labels
        for k in range(n_regimes):
            out[f"regime_prob_{k}"] = (labels == k).astype(float)
    return out


def fit_pca_projection(train_df: pd.DataFrame, feature_cols: Sequence[str], n_components: int = 2) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )
    pipe.fit(train_df[list(feature_cols)])
    return pipe


# ---------------------------------------------------------------------------
# Experiment 5: Adaptive contextual decision policy
# ---------------------------------------------------------------------------

def _fit_return_model(train: pd.DataFrame, feature_cols: Sequence[str], model_type: str = "hgb", random_state: int = 42) -> Pipeline:
    if model_type == "hgb":
        reg = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.04, l2_regularization=0.02, random_state=random_state)
    elif model_type == "rf":
        reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError("model_type must be 'hgb' or 'rf'")
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", reg),
        ]
    )
    pipe.fit(train[list(feature_cols)], train["continue_return"].astype(float))
    return pipe


def choose_actions_from_expected_returns(
    df: pd.DataFrame,
    expected_continue: np.ndarray,
    cost_bps: float = 5.0,
    min_edge_bps: float = 2.0,
) -> pd.DataFrame:
    """Convert expected continue return into continue/reverse/abstain actions."""
    out = df.copy()
    cost = cost_bps / 10_000.0
    min_edge = min_edge_bps / 10_000.0

    out["expected_continue_gross"] = expected_continue
    out["expected_reverse_gross"] = -expected_continue
    out["expected_continue_net"] = out["expected_continue_gross"] - cost
    out["expected_reverse_net"] = out["expected_reverse_gross"] - cost

    best_is_continue = out["expected_continue_net"] >= out["expected_reverse_net"]
    best_edge = np.where(best_is_continue, out["expected_continue_net"], out["expected_reverse_net"])
    out["expected_best_edge"] = best_edge
    out["action"] = "ABSTAIN"
    out.loc[(best_edge > min_edge) & best_is_continue, "action"] = "CONTINUE"
    out.loc[(best_edge > min_edge) & (~best_is_continue), "action"] = "REVERSE"

    out["realized_return"] = 0.0
    m_continue = out["action"] == "CONTINUE"
    m_reverse = out["action"] == "REVERSE"
    out.loc[m_continue, "realized_return"] = out.loc[m_continue, "continue_return"] - cost
    out.loc[m_reverse, "realized_return"] = out.loc[m_reverse, "reverse_return"] - cost
    out["traded"] = out["action"].isin(["CONTINUE", "REVERSE"]).astype(int)
    return out


def walk_forward_adaptive_policy(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_days: int = 60,
    test_days: int = 10,
    step_days: Optional[int] = None,
    n_regimes: int = 5,
    regime_method: str = "gmm",
    model_type: str = "rf",
    cost_bps: float = 5.0,
    min_edge_bps: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Walk-forward train/test backtest with latent regimes fitted on train only."""
    if step_days is None:
        step_days = test_days
    df = model_df.copy().sort_values(["date", "symbol"])
    df["date"] = _to_date(df["date"])
    dates = np.array(sorted(df["date"].unique()))
    outputs = []

    if len(dates) < train_days + test_days:
        raise ValueError(f"Not enough dates: have {len(dates)}, need at least {train_days + test_days}")

    for start in range(0, len(dates) - train_days - test_days + 1, step_days):
        train_dates = dates[start : start + train_days]
        test_dates = dates[start + train_days : start + train_days + test_days]
        train = df[df["date"].isin(train_dates)].copy()
        test = df[df["date"].isin(test_dates)].copy()
        if train.empty or test.empty:
            continue

        # Fit unsupervised regimes on train only, then add regime probabilities to both sets.
        regime_model = fit_latent_regime_model(train, feature_cols, n_regimes, regime_method, random_state)
        train_r = add_latent_regime_features(train, regime_model, feature_cols, n_regimes)
        test_r = add_latent_regime_features(test, regime_model, feature_cols, n_regimes)

        regime_cols = [f"regime_prob_{k}" for k in range(n_regimes)]
        final_features = list(feature_cols) + regime_cols

        return_model = _fit_return_model(train_r, final_features, model_type, random_state)
        pred_continue = return_model.predict(test_r[final_features])
        pred = choose_actions_from_expected_returns(test_r, pred_continue, cost_bps, min_edge_bps)
        pred["fold_start"] = pd.Timestamp(train_dates[0])
        pred["fold_train_end"] = pd.Timestamp(train_dates[-1])
        pred["fold_test_start"] = pd.Timestamp(test_dates[0])
        pred["fold_test_end"] = pd.Timestamp(test_dates[-1])
        outputs.append(pred)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def baseline_policy(events: pd.DataFrame, action: str, cost_bps: float = 5.0) -> pd.DataFrame:
    """Always-continue, always-reverse, or always-abstain baseline."""
    if action not in {"CONTINUE", "REVERSE", "ABSTAIN"}:
        raise ValueError("action must be CONTINUE, REVERSE, or ABSTAIN")
    out = events.copy()
    cost = cost_bps / 10_000.0
    out["action"] = action
    out["realized_return"] = 0.0
    if action == "CONTINUE":
        out["realized_return"] = out["continue_return"] - cost
    elif action == "REVERSE":
        out["realized_return"] = out["reverse_return"] - cost
    out["traded"] = int(action != "ABSTAIN")
    return out


def daily_portfolio_returns(policy_output: pd.DataFrame, use_active_trades_only: bool = False) -> pd.Series:
    df = policy_output.copy()
    df["date"] = _to_date(df["date"])
    if use_active_trades_only:
        traded = df[df["traded"] == 1]
        daily = traded.groupby("date")["realized_return"].mean()
        all_dates = pd.Index(sorted(df["date"].unique()), name="date")
        return daily.reindex(all_dates, fill_value=0.0).sort_index()
    return df.groupby("date")["realized_return"].mean().sort_index()


def summarize_strategy(policy_output: pd.DataFrame, use_active_trades_only: bool = False) -> Dict[str, float]:
    daily = daily_portfolio_returns(policy_output, use_active_trades_only=use_active_trades_only)
    if daily.empty:
        return {}
    equity = (1.0 + daily).cumprod()
    vol = daily.std(ddof=1)
    sharpe = np.sqrt(252.0) * daily.mean() / vol if vol and vol > 0 else np.nan
    return {
        "n_dates": float(daily.shape[0]),
        "n_events": float(policy_output.shape[0]),
        "n_trades": float(policy_output["traded"].sum()) if "traded" in policy_output else np.nan,
        "trade_rate": float(policy_output["traded"].mean()) if "traded" in policy_output else np.nan,
        "mean_daily_return": float(daily.mean()),
        "median_daily_return": float(daily.median()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "annualized_sharpe": float(sharpe),
        "max_drawdown": max_drawdown(equity),
        "daily_win_rate": float((daily > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Experiment 6: Monte Carlo and scenario robustness
# ---------------------------------------------------------------------------

def date_block_bootstrap(
    daily_returns: pd.Series,
    n_sims: int = 10_000,
    horizon_days: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap full dates from a daily strategy return series."""
    rng = np.random.default_rng(seed)
    r = pd.Series(daily_returns).dropna().astype(float).to_numpy()
    if len(r) == 0:
        raise ValueError("daily_returns is empty")
    if horizon_days is None:
        horizon_days = len(r)

    rows = []
    for sim in range(n_sims):
        sampled = rng.choice(r, size=horizon_days, replace=True)
        equity = np.cumprod(1.0 + sampled)
        vol = sampled.std(ddof=1)
        rows.append(
            {
                "simulation": sim,
                "cumulative_return": equity[-1] - 1.0,
                "mean_daily_return": sampled.mean(),
                "annualized_sharpe": np.sqrt(252.0) * sampled.mean() / vol if vol > 0 else np.nan,
                "max_drawdown": max_drawdown(equity),
                "win_rate": (sampled > 0).mean(),
                "cvar_5pct_daily": sampled[sampled <= np.quantile(sampled, 0.05)].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_monte_carlo(mc: pd.DataFrame) -> Dict[str, float]:
    return {
        "prob_positive_return": float((mc["cumulative_return"] > 0).mean()),
        "median_cumulative_return": float(mc["cumulative_return"].median()),
        "p05_cumulative_return": float(mc["cumulative_return"].quantile(0.05)),
        "p95_cumulative_return": float(mc["cumulative_return"].quantile(0.95)),
        "median_sharpe": float(mc["annualized_sharpe"].median()),
        "median_max_drawdown": float(mc["max_drawdown"].median()),
        "prob_drawdown_worse_5pct": float((mc["max_drawdown"] < -0.05).mean()),
    }


def simulate_parkinson_barriers(
    entry_price: float,
    direction: int,
    parkinson_vol_annual: float,
    tp_pct: float = 0.01,
    sl_pct: float = 0.005,
    horizon_days: int = 1,
    steps_per_day: int = 78,
    n_sims: int = 10_000,
    drift_annual: float = 0.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Monte Carlo TP/SL hit probabilities using range-based volatility.

    direction = +1 means long; direction = -1 means short. The process is simulated
    in intraday substeps. It is a stress-testing tool, not an execution simulator.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / (252.0 * steps_per_day)
    n_steps = horizon_days * steps_per_day
    sigma = float(parkinson_vol_annual)
    mu = float(drift_annual)

    z = rng.standard_normal((n_sims, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    prices = entry_price * np.exp(np.cumsum(log_returns, axis=1))

    if direction == 1:
        tp = entry_price * (1.0 + tp_pct)
        sl = entry_price * (1.0 - sl_pct)
        hit_tp = prices >= tp
        hit_sl = prices <= sl
    elif direction == -1:
        tp = entry_price * (1.0 - tp_pct)
        sl = entry_price * (1.0 + sl_pct)
        hit_tp = prices <= tp
        hit_sl = prices >= sl
    else:
        raise ValueError("direction must be +1 or -1")

    first_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), np.inf)
    first_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), np.inf)
    wins = first_tp < first_sl
    losses = first_sl < first_tp
    neither = ~(wins | losses)

    final_price = prices[:, -1]
    final_return = direction * (final_price / entry_price - 1.0)
    return {
        "win_probability": float(wins.mean()),
        "loss_probability": float(losses.mean()),
        "neither_probability": float(neither.mean()),
        "mean_final_directional_return": float(final_return.mean()),
        "p05_final_directional_return": float(np.quantile(final_return, 0.05)),
        "p95_final_directional_return": float(np.quantile(final_return, 0.95)),
    }


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------

def run_smoke_test(seed: int = 42) -> Dict[str, Any]:
    """Run the full pipeline on synthetic data to verify the implementation."""
    cfg = PipelineConfig(random_state=seed, overreaction_windows=(20, 40), event_memory_windows=(10, 20), n_latent_regimes=4)
    intraday, daily = make_synthetic_intraday_data(n_symbols=60, n_days=90, seed=seed)
    events, all_days = compute_opening_events(
        intraday,
        opening_start=cfg.opening_start,
        decision_time=cfg.decision_time,
        extreme_quantile=cfg.extreme_quantile,
        filter_extreme=True,
    )
    model_df = assemble_model_frame(events, all_days, daily, cfg)
    feature_cols = default_feature_columns(model_df)
    preds = walk_forward_adaptive_policy(
        model_df,
        feature_cols=feature_cols,
        train_days=45,
        test_days=10,
        n_regimes=cfg.n_latent_regimes,
        cost_bps=cfg.cost_bps,
        min_edge_bps=cfg.min_edge_bps,
        random_state=seed,
        model_type="rf",
    )
    adaptive_summary = summarize_strategy(preds)
    always_reverse = baseline_policy(model_df[model_df["date"].isin(preds["date"].unique())], "REVERSE", cfg.cost_bps)
    reverse_summary = summarize_strategy(always_reverse)
    daily_adaptive = daily_portfolio_returns(preds)
    mc = date_block_bootstrap(daily_adaptive, n_sims=1000, seed=seed)
    return {
        "n_intraday_rows": intraday.shape[0],
        "n_daily_rows": daily.shape[0],
        "n_events": events.shape[0],
        "n_features": len(feature_cols),
        "adaptive_summary": adaptive_summary,
        "always_reverse_summary": reverse_summary,
        "mc_summary": summarize_monte_carlo(mc),
    }
