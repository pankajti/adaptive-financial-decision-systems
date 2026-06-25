from __future__ import annotations

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from numpy.ma.core import append
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icaif_2026.lab.orb.db_connections import engine


FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "vol_5d",
    "vol_20d",
    "natr_14",
    "volume_spike",
    "gap_pct",
    "close_position",
    "avg_dollar_volume_20",
    "sector_relative_return_20d",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily features available after the close of each date."""

    required = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sector",
        "industry",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    symbol_group = df.groupby("symbol", sort=False)

    # Returns
    df["return_1d"] = symbol_group["close"].pct_change()

    df["return_5d"] = (
        df["close"] / symbol_group["close"].shift(5) - 1
    )

    df["return_20d"] = (
        df["close"] / symbol_group["close"].shift(20) - 1
    )

    # Realized volatility
    df["vol_5d"] = (
        df.groupby("symbol", sort=False)["return_1d"]
        .transform(lambda x: x.rolling(5, min_periods=5).std())
    )

    df["vol_20d"] = (
        df.groupby("symbol", sort=False)["return_1d"]
        .transform(lambda x: x.rolling(20, min_periods=20).std())
    )

    # True range and ATR
    previous_close = symbol_group["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["true_range"] = true_range

    df["atr_14"] = (
        df.groupby("symbol", sort=False)["true_range"]
        .transform(lambda x: x.rolling(14, min_periods=14).mean())
    )

    df["natr_14"] = df["atr_14"] / df["close"]

    # Liquidity
    df["dollar_volume"] = df["close"] * df["volume"]

    df["avg_dollar_volume_20"] = (
        df.groupby("symbol", sort=False)["dollar_volume"]
        .transform(lambda x: x.rolling(20, min_periods=20).mean())
    )

    average_volume_20 = (
        df.groupby("symbol", sort=False)["volume"]
        .transform(lambda x: x.rolling(20, min_periods=20).mean())
    )

    df["volume_spike"] = (
        df["volume"] / average_volume_20.replace(0, np.nan)
    )

    # Overnight gap known on date t and used after date-t close
    df["gap_pct"] = df["open"] / previous_close - 1

    # Closing location within the daily range
    daily_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["close_position"] = (
        (df["close"] - df["low"]) / daily_range
    )

    return df


def add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leave-one-out sector-relative 20-day momentum."""

    df = df.copy()

    sector_group = df.groupby(
        ["date", "sector"],
        dropna=False,
        sort=False,
    )["return_20d"]

    sector_sum = sector_group.transform("sum")
    sector_count = sector_group.transform("count")

    denominator = (sector_count - 1).replace(0, np.nan)

    df["sector_return_20d_ex_stock"] = (
        sector_sum - df["return_20d"]
    ) / denominator

    df["sector_relative_return_20d"] = (
        df["return_20d"] -
        df["sector_return_20d_ex_stock"]
    )

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preliminary proxy target.

    This is next-session close-to-close return, not an ORB return.
    """

    df = df.copy()

    next_close = (
        df.groupby("symbol", sort=False)["close"]
        .shift(-1)
    )

    df["next_day_return"] = next_close / df["close"] - 1

    df["target"] = np.where(
        df["next_day_return"].notna(),
        (df["next_day_return"] > 0).astype(int),
        np.nan,
    )

    return df


def rank_candidates(
    df: pd.DataFrame,
    score_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Select the top-N eligible instruments for each date."""

    required = {"date", score_col, "next_day_return"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")

    eligible = df.dropna(
        subset=[score_col, "next_day_return"]
    ).copy()

    return (
        eligible
        .sort_values(
            ["date", score_col, "symbol"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=False)
        .head(top_n)
        .copy()
    )


def maximum_drawdown(daily_returns: pd.Series) -> float:
    equity_curve = (1 + daily_returns.fillna(0)).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return float(drawdown.min())


def evaluate_strategy(ranked_df: pd.DataFrame) -> dict[str, float]:
    """Evaluate an equal-weighted top-N daily selection portfolio."""

    daily_returns = (
        ranked_df.groupby("date")["next_day_return"]
        .mean()
        .sort_index()
        .dropna()
    )

    if daily_returns.empty:
        raise ValueError("No daily returns available for evaluation.")

    daily_volatility = daily_returns.std(ddof=1)

    sharpe = (
        np.sqrt(252) * daily_returns.mean() / daily_volatility
        if daily_volatility > 0
        else np.nan
    )

    cumulative_return = (1 + daily_returns).prod() - 1

    return {
        "number_of_days": len(daily_returns),
        "avg_daily_return": daily_returns.mean(),
        "annualized_volatility": daily_volatility * np.sqrt(252),
        "annualized_sharpe": sharpe,
        "cumulative_return": cumulative_return,
        "maximum_drawdown": maximum_drawdown(daily_returns),
        "daily_win_rate": (daily_returns > 0).mean(),
    }


def load_history() -> pd.DataFrame:
    query = """
        SELECT
            i.symbol,
            db.date,
            db.open,
            db.high,
            db.low,
            db.close,
            db.volume,
            i.sector,
            i.industry
        FROM daily_bars AS db
        JOIN instruments AS i
          ON i.id = db.instrument_id
        WHERE i.market = 'US'
          AND i.is_active = TRUE
          AND db.date >= '2019-01-01'
        ORDER BY i.symbol, db.date
    """

    history = pd.read_sql(query, con=engine)
    history["date"] = pd.to_datetime(history["date"])

    return history


def main() -> None:
    history = load_history()

    features = add_features(history)
    features = add_sector_relative_features(features)
    features = create_target(features)

    # Remove rows without a valid forward target.
    model_data = features.dropna(
        subset=["target", "next_day_return"]
    ).copy()

    train_end = pd.Timestamp("2025-12-31")

    train_df = model_data[
        model_data["date"] <= train_end
    ].copy()

    test_df = model_data[
        model_data["date"] > train_end
    ].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].astype(int)

    X_test = test_df[FEATURE_COLS]

    # Logistic-regression baseline
    logistic_model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)

    test_df["lr_score"] = logistic_model.predict_proba(
        X_test
    )[:, 1]

    # Nonlinear model
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    lgbm_model.fit(X_train, y_train)

    test_df["lgbm_score"] = lgbm_model.predict_proba(
        X_test
    )[:, 1]

    top_n = 10

    top_volume = rank_candidates(
        test_df,
        "avg_dollar_volume_20",
        top_n=top_n,
    )

    top_atr = rank_candidates(
        test_df,
        "natr_14",
        top_n=top_n,
    )

    top_logistic = rank_candidates(
        test_df,
        "lr_score",
        top_n=top_n,
    )

    top_lgbm = rank_candidates(
        test_df,
        "lgbm_score",
        top_n=top_n,
    )

    results = pd.DataFrame(
        {
            "TopDollarVolume": evaluate_strategy(top_volume),
            "TopNATR": evaluate_strategy(top_atr),
            "Logistic": evaluate_strategy(top_logistic),
            "LightGBM": evaluate_strategy(top_lgbm),
        }
    ).T

    print("\nModel period")
    print(
        {
            "train_start": train_df["date"].min(),
            "train_end": train_df["date"].max(),
            "test_start": test_df["date"].min(),
            "test_end": test_df["date"].max(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        }
    )

    print("\nStrategy comparison")
    results['train_start']=train_df["date"].min()
    results['train_end']=train_df["date"].max()
    results['test_start']=test_df["date"].min()
    results['test_end']=test_df["date"].max()
    results.to_csv("results.csv", mode="a")

    print(results.round(6).head())


if __name__ == "__main__":
    main()