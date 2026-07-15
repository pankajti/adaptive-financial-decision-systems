"""Agri ETF Adaptive Market-State Learning package."""
from .config import ResearchConfig, RANDOM_STATE
from .actions import ACTION_COLS, ACTIONS, add_event_and_action_returns, prepare_model_frame
from .data import load_data, normalize_ohlcv_columns, load_from_yfinance, make_synthetic_agri_etf_data
from .features import build_model_base, get_feature_columns, add_symbol_features, add_cross_etf_context, BASE_FEATURE_COLS
from .projections import fit_exploratory_pca, add_pca_regimes, make_augmented_features
from .policy import walk_forward_adaptive_policy
from .evaluation import daily_portfolio_returns, summarize_daily_returns, fixed_action_summary, action_distribution, regime_interpretation
from .stress import run_cost_sensitivity, block_bootstrap_paths, monte_carlo_summary
