"""Configuration for the Agri ETF Adaptive Market-State Learning project."""
from __future__ import annotations

from dataclasses import dataclass

RANDOM_STATE = 42


@dataclass
class ResearchConfig:
    """Top-level experiment configuration.

    DATA_MODE can be:
    - "synthetic": runnable offline using simulated agri-ETF-like OHLCV data.
    - "yfinance": downloads ticker data via yfinance.
    - "csv": reads LOCAL_CSV_PATH with date/symbol/open/high/low/close/volume fields.
    """

    DATA_MODE: str = "synthetic"
    LOCAL_CSV_PATH: str = "agri_etf_ohlcv.csv"
    START_DATE: str = "2016-01-01"
    END_DATE: str | None = None

    TRADE_TICKERS: tuple[str, ...] = ("DBA", "WEAT", "CORN", "SOYB", "CANE")
    CONTEXT_TICKERS: tuple[str, ...] = ("DBC", "UUP", "USO")

    HORIZON_DAYS: int = 5
    EVENT_ONLY: bool = True
    DISLOCATION_Z_THRESHOLD: float = 1.5
    MOMENTUM_Z_THRESHOLD: float = 1.0
    RANGE_SHOCK_THRESHOLD: float = 1.5

    TRAIN_DAYS: int = 252
    TEST_DAYS: int = 126
    STEP_DAYS: int = 126
    N_PCA_COMPONENTS: int = 3
    N_CLUSTERS: int = 4
    MIN_EDGE_BPS: float = 2.0
    COST_BPS: float = 5.0
    RUN_VAE: bool = False
    RUN_COST_SENSITIVITY: bool = False
    SYNTHETIC_PERIODS: int = 900
