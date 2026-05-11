"""LightGBM corners model — separate Poisson regressors for home and away corners.

The total over/under probabilities are derived by simulating independent Poisson
draws from the predicted means.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_corners_features, feature_columns, impute_league_defaults


_DEFAULT_LINES: tuple[float, ...] = (7.5, 8.5, 9.5, 10.5, 11.5)


@dataclass
class CornersModel:
    home_model: lgb.Booster
    away_model: lgb.Booster
    features: list[str]
    windows: tuple[int, ...]


def _train_one(X: pd.DataFrame, y: pd.Series, *, num_boost_round: int = 200) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y)
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def fit_corners(df: pd.DataFrame, *, windows: tuple[int, ...] = (5, 10)) -> CornersModel:
    """Fit home/away corners models on completed matches in df."""
    feat = build_corners_features(df, windows=windows)
    cols = feature_columns(windows)
    X = impute_league_defaults(feat[cols], windows=windows)
    yh = feat["home_corners"].astype(float)
    ya = feat["away_corners"].astype(float)
    home_m = _train_one(X, yh)
    away_m = _train_one(X, ya)
    return CornersModel(home_model=home_m, away_model=away_m, features=cols, windows=windows)


def _predict_means(
    model: CornersModel, df_pred: pd.DataFrame, df_history: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Predict (lambda_home, lambda_away) corners means.

    Builds a fresh feature frame from `df_history + df_pred`, tagging each prediction
    row with a synthetic match_id so we can recover its features after sorting.
    """
    hist = df_history.copy().assign(match_id=[f"H_{i}" for i in range(len(df_history))])
    pred = df_pred.copy().assign(match_id=[f"P_{i}" for i in range(len(df_pred))])
    combined = pd.concat([hist, pred], ignore_index=True).sort_values("date", kind="stable")
    feat = build_corners_features(combined, windows=model.windows)
    X = impute_league_defaults(feat[model.features], windows=model.windows)
    # Recover prediction rows in original df_pred order
    pred_mask = feat["match_id"].astype(str).str.startswith("P_")
    sub = feat.loc[pred_mask].copy()
    sub["_order"] = sub["match_id"].map(lambda s: int(s.split("_", 1)[1]))
    sub = sub.sort_values("_order")
    X_pred = X.loc[sub.index]
    lam_h = model.home_model.predict(X_pred)
    lam_a = model.away_model.predict(X_pred)
    return np.asarray(lam_h), np.asarray(lam_a)


def predict_corners(
    model: CornersModel,
    df_history: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    lines: Iterable[float] = _DEFAULT_LINES,
    n_sim: int = 30_000,
    rng_seed: int = 7,
) -> pd.DataFrame:
    """For each row in df_pred, return predicted means + P(total > line) for each line.

    Independent Poisson simulation is used (home & away corner counts can mildly
    correlate in reality, but for unconditional totals the bias is small and lets us
    keep the model simple).
    """
    lam_h, lam_a = _predict_means(model, df_pred, df_history)
    rng = np.random.default_rng(rng_seed)
    sims_h = rng.poisson(lam=np.tile(lam_h[:, None], (1, n_sim)))
    sims_a = rng.poisson(lam=np.tile(lam_a[:, None], (1, n_sim)))
    total = sims_h + sims_a

    out = pd.DataFrame(index=df_pred.index)
    out["lam_corners_home"] = lam_h
    out["lam_corners_away"] = lam_a
    out["lam_corners_total"] = lam_h + lam_a
    for L in lines:
        out[f"p_corners_over_{L}"] = (total > L).mean(axis=1)
        out[f"p_corners_under_{L}"] = 1.0 - out[f"p_corners_over_{L}"]
    return out
