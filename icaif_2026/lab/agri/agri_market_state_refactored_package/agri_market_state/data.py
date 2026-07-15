"""Data loading utilities for agri ETF OHLCV panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, ResearchConfig


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe to date/symbol/open/high/low/close/volume columns."""
    df = df.copy()
    rename_map: dict[str, str] = {}
    for c in df.columns:
        lc = str(c).lower().strip().replace(" ", "_")
        if lc in {"date", "datetime", "timestamp"}:
            rename_map[c] = "date"
        elif lc in {"symbol", "ticker"}:
            rename_map[c] = "symbol"
        elif lc in {"open", "open_price"}:
            rename_map[c] = "open"
        elif lc in {"high", "high_price"}:
            rename_map[c] = "high"
        elif lc in {"low", "low_price"}:
            rename_map[c] = "low"
        elif lc in {"close", "adj_close", "adjusted_close", "close_price"}:
            rename_map[c] = "close"
        elif lc in {"volume", "vol"}:
            rename_map[c] = "volume"

    df = df.rename(columns=rename_map)
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_from_yfinance(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV data with yfinance and normalize it."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance or switch DATA_MODE to synthetic/csv.") from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)
    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            tmp = raw[ticker].copy()
            tmp = tmp.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            tmp["date"] = tmp.index
            tmp["symbol"] = ticker
            rows.append(tmp[["date", "symbol", "open", "high", "low", "close", "volume"]])
    else:
        tmp = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        tmp["date"] = tmp.index
        tmp["symbol"] = tickers[0]
        rows.append(tmp[["date", "symbol", "open", "high", "low", "close", "volume"]])

    if not rows:
        raise ValueError("No data returned. Check tickers/date range/internet.")
    return normalize_ohlcv_columns(pd.concat(rows, ignore_index=True))


def make_synthetic_agri_etf_data(
    tickers: list[str], start: str = "2016-01-01", periods: int = 1800, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Create synthetic agri-ETF-like OHLCV data for offline testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    n = len(dates)
    regimes = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])
    agri = rng.normal(0, 0.006, n)
    grain = rng.normal(0, 0.005, n)
    soft = rng.normal(0, 0.006, n)
    dollar = rng.normal(0, 0.0035, n)
    energy = rng.normal(0, 0.006, n)
    for i in range(1, n):
        if regimes[i] == 1:
            agri[i] += 0.25 * agri[i - 1]
            grain[i] += 0.20 * grain[i - 1]
        elif regimes[i] == 2:
            agri[i] -= 0.30 * agri[i - 1]
            grain[i] -= 0.25 * grain[i - 1]

    loadings = {
        "DBA": (0.75, 0.25, 0.25, -0.20, 0.10, 0.006),
        "WEAT": (0.55, 0.75, 0.05, -0.15, 0.05, 0.010),
        "CORN": (0.60, 0.70, 0.05, -0.15, 0.10, 0.009),
        "SOYB": (0.60, 0.65, 0.05, -0.12, 0.12, 0.009),
        "CANE": (0.45, 0.05, 0.80, -0.10, 0.15, 0.011),
        "DBC": (0.35, 0.20, 0.15, -0.25, 0.55, 0.007),
        "UUP": (-0.15, -0.05, -0.05, 1.00, -0.05, 0.003),
        "USO": (0.15, 0.05, 0.10, -0.20, 1.00, 0.014),
    }
    rows = []
    for ticker in tickers:
        l = loadings.get(ticker, (0.5, 0.3, 0.2, -0.1, 0.1, 0.008))
        eps = rng.normal(0, l[5], n)
        ret = l[0] * agri + l[1] * grain + l[2] * soft + l[3] * dollar + l[4] * energy + eps
        shock_idx = rng.choice(np.arange(30, n - 30), size=max(5, n // 90), replace=False)
        ret[shock_idx] += rng.normal(0, 0.035, len(shock_idx))
        close = 25 * np.exp(np.cumsum(ret))
        open_ = close / (1 + ret) * (1 + rng.normal(0, 0.002, n))
        intraday_range = np.abs(rng.normal(0.012, 0.006, n)) + np.abs(ret) * 0.6
        high = np.maximum(open_, close) * (1 + intraday_range / 2)
        low = np.minimum(open_, close) * (1 - intraday_range / 2)
        volume = rng.lognormal(mean=12.0, sigma=0.55, size=n).astype(int)
        rows.append(pd.DataFrame({"date": dates, "symbol": ticker, "open": open_, "high": high, "low": low, "close": close, "volume": volume}))
    return normalize_ohlcv_columns(pd.concat(rows, ignore_index=True))


def load_data(cfg: ResearchConfig) -> pd.DataFrame:
    """Load the configured OHLCV panel."""
    tickers = sorted(set(cfg.TRADE_TICKERS + cfg.CONTEXT_TICKERS))
    if cfg.DATA_MODE == "synthetic":
        return make_synthetic_agri_etf_data(tickers, cfg.START_DATE, periods=cfg.SYNTHETIC_PERIODS, seed=RANDOM_STATE)
    if cfg.DATA_MODE == "yfinance":
        return load_from_yfinance(tickers, cfg.START_DATE, cfg.END_DATE)
    if cfg.DATA_MODE == "csv":
        return normalize_ohlcv_columns(pd.read_csv(cfg.LOCAL_CSV_PATH))
    raise ValueError(f"Unknown DATA_MODE: {cfg.DATA_MODE}")
