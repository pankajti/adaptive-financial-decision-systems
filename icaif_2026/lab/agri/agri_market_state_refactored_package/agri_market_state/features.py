"""Feature engineering for agri ETF market-state learning."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Simple rolling RSI implementation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_symbol_features(df: pd.DataFrame, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    """Add trend, momentum, mean-reversion, volatility and forward-return features by symbol."""
    parts = []
    for _, x in df.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        x = x.copy().sort_values("date")
        close, open_, high, low, vol = x["close"], x["open"], x["high"], x["low"], x["volume"]
        prev_close = close.shift(1)
        x["daily_return"] = close.pct_change()
        x["log_return"] = np.log(close / prev_close)
        next_open = open_.shift(-1)
        for h in horizons:
            x[f"fwd_return_{h}d"] = close.shift(-h) / next_open - 1

        # Trend features
        x["sma_10"] = close.rolling(10, min_periods=10).mean()
        x["sma_20"] = close.rolling(20, min_periods=20).mean()
        x["sma_60"] = close.rolling(60, min_periods=60).mean()
        x["trend_score"] = x["sma_20"] / x["sma_60"] - 1
        x["trend_slope_20"] = x["sma_20"].pct_change(5)
        x["price_vs_sma60"] = close / x["sma_60"] - 1
        x["ret_20"] = close / close.shift(20) - 1
        x["ret_60"] = close / close.shift(60) - 1

        # Momentum features
        x["ret_5"] = close / close.shift(5) - 1
        x["ret_10"] = close / close.shift(10) - 1
        vol_20 = x["daily_return"].rolling(20, min_periods=20).std()
        x["momentum_score"] = x["ret_5"] / vol_20.replace(0, np.nan)
        x["momentum_20_score"] = x["ret_20"] / (x["daily_return"].rolling(60, min_periods=40).std().replace(0, np.nan) * np.sqrt(20))

        # Mean-reversion / dislocation features
        mean20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        mean60 = close.rolling(60, min_periods=40).mean()
        std60 = close.rolling(60, min_periods=40).std()
        x["dislocation_zscore_20"] = (close - mean20) / std20.replace(0, np.nan)
        x["dislocation_zscore_60"] = (close - mean60) / std60.replace(0, np.nan)
        x["rsi_14"] = rsi(close, 14)
        x["rsi_centered"] = (x["rsi_14"] - 50) / 50

        # Volatility/range/liquidity features
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        x["atr_14_pct"] = tr.rolling(14, min_periods=14).mean() / close
        x["range_pct"] = (high - low) / open_.replace(0, np.nan)
        x["range_shock_20"] = x["range_pct"] / x["range_pct"].shift(1).rolling(20, min_periods=15).median()
        hl_log_sq = np.log(high / low).pow(2)
        x["parkinson_vol_20"] = np.sqrt(hl_log_sq.rolling(20, min_periods=20).mean() / (4 * np.log(2)))
        x["parkinson_vol_60"] = np.sqrt(hl_log_sq.rolling(60, min_periods=40).mean() / (4 * np.log(2)))
        x["volume_zscore_20"] = (np.log1p(vol) - np.log1p(vol).rolling(20, min_periods=20).mean()) / np.log1p(vol).rolling(20, min_periods=20).std()
        rolling_high_60 = close.rolling(60, min_periods=40).max()
        rolling_low_60 = close.rolling(60, min_periods=40).min()
        x["drawdown_60"] = close / rolling_high_60 - 1
        x["distance_from_60d_low"] = close / rolling_low_60 - 1

        # Directional actions available to the policy.
        x["trend_dir"] = np.sign(x["trend_score"])
        x["momentum_dir"] = np.sign(x["momentum_score"])
        x["mean_reversion_dir"] = -np.sign(x["dislocation_zscore_20"])
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def add_cross_etf_context(df: pd.DataFrame, trade_tickers: tuple[str, ...], context_tickers: tuple[str, ...]) -> pd.DataFrame:
    """Add cross-ETF market context such as basket return and dispersion."""
    ret_wide = df.pivot(index="date", columns="symbol", values="daily_return").sort_index()
    trade_cols = [c for c in trade_tickers if c in ret_wide.columns]
    ctx_cols = [c for c in context_tickers if c in ret_wide.columns]
    context = pd.DataFrame(index=ret_wide.index)
    context["agri_basket_return"] = ret_wide[trade_cols].mean(axis=1)
    context["agri_cross_dispersion"] = ret_wide[trade_cols].std(axis=1)
    context["agri_positive_breadth"] = (ret_wide[trade_cols] > 0).mean(axis=1)
    context["agri_abs_move_median"] = ret_wide[trade_cols].abs().median(axis=1)
    for c in ctx_cols:
        context[f"ctx_{c}_return"] = ret_wide[c]
        context[f"ctx_{c}_return_5d"] = ret_wide[c].rolling(5, min_periods=5).sum()
        context[f"ctx_{c}_vol_20"] = ret_wide[c].rolling(20, min_periods=20).std()
    return df.merge(context.reset_index(), on="date", how="left")


BASE_FEATURE_COLS = [
    "trend_score", "trend_slope_20", "price_vs_sma60", "ret_20", "ret_60",
    "ret_5", "ret_10", "momentum_score", "momentum_20_score",
    "dislocation_zscore_20", "dislocation_zscore_60", "rsi_centered",
    "atr_14_pct", "range_pct", "range_shock_20", "parkinson_vol_20", "parkinson_vol_60",
    "volume_zscore_20", "drawdown_60", "distance_from_60d_low",
    "agri_basket_return", "agri_cross_dispersion", "agri_positive_breadth", "agri_abs_move_median",
]


def build_model_base(raw_df: pd.DataFrame, trade_tickers: tuple[str, ...], context_tickers: tuple[str, ...], horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    """Create the feature dataframe and retain only tradable agri ETF symbols."""
    feature_df = add_symbol_features(raw_df, horizons=horizons)
    feature_df = add_cross_etf_context(feature_df, trade_tickers, context_tickers)
    return feature_df[feature_df["symbol"].isin(trade_tickers)].copy()


def get_feature_columns(model_df: pd.DataFrame, base_cols: list[str] | None = None) -> list[str]:
    """Return feature columns available in a model dataframe."""
    base_cols = BASE_FEATURE_COLS if base_cols is None else base_cols
    context_feature_cols = [c for c in model_df.columns if c.startswith("ctx_")]
    return [c for c in list(base_cols) + context_feature_cols if c in model_df.columns]
