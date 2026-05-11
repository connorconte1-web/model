"""Feature engineering for the ML corners model.

Rolling per-team rates are *strictly causal* (uses only matches with date < this match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_LEAGUE_DEFAULTS = {
    "corners_for": 5.0,
    "corners_against": 5.0,
    "shots_for": 12.0,
    "shots_against": 12.0,
    "sot_for": 4.0,
    "sot_against": 4.0,
}


def _per_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-rows (home perspective + away perspective).

    `df` must already have a `match_id` column with stable per-match identifiers.
    """
    home = pd.DataFrame(
        {
            "date": df["date"].values,
            "team": df["home"].values,
            "opp": df["away"].values,
            "venue": "H",
            "corners_for": df["home_corners"].values,
            "corners_against": df["away_corners"].values,
            "shots_for": df["home_shots"].values,
            "shots_against": df["away_shots"].values,
            "sot_for": df["home_shots_on_target"].values,
            "sot_against": df["away_shots_on_target"].values,
            "match_id": df["match_id"].values,
        }
    )
    away = pd.DataFrame(
        {
            "date": df["date"].values,
            "team": df["away"].values,
            "opp": df["home"].values,
            "venue": "A",
            "corners_for": df["away_corners"].values,
            "corners_against": df["home_corners"].values,
            "shots_for": df["away_shots"].values,
            "shots_against": df["home_shots"].values,
            "sot_for": df["away_shots_on_target"].values,
            "sot_against": df["home_shots_on_target"].values,
            "match_id": df["match_id"].values,
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")


def _rolling_features(long_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return causal rolling means per team (shifted by 1 to exclude the match itself)."""
    g = long_df.groupby("team", sort=False)
    out = pd.DataFrame(index=long_df.index)
    for col in ["corners_for", "corners_against", "shots_for", "shots_against", "sot_for", "sot_against"]:
        rolled = g[col].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        # apply returns a series indexed by (team, original_index) — drop level
        if isinstance(rolled.index, pd.MultiIndex):
            rolled = rolled.reset_index(level=0, drop=True)
        out[f"{col}_r{window}"] = rolled.values
    return out


def build_corners_features(df: pd.DataFrame, *, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Build a per-match feature frame for the corners model.

    Returns one row per match with home_/away_ rolling rates as separate columns plus
    pre-match xG. Target columns `home_corners` and `away_corners` are preserved.
    The output preserves df's row order; rows do *not* carry a match_id column to
    callers (it's internal).
    """
    base = df.copy()
    if "match_id" not in base.columns:
        base = base.assign(match_id=np.arange(len(base)))
    long_df = _per_team_long(base).reset_index(drop=True)
    rolled_parts = [_rolling_features(long_df, w) for w in windows]
    feat = pd.concat([long_df[["match_id", "team", "venue"]]] + rolled_parts, axis=1)

    # Split back into home/away rows and merge on match_id
    home_feat = feat[feat["venue"] == "H"].drop(columns=["venue", "team"]).add_prefix("home_")
    home_feat = home_feat.rename(columns={"home_match_id": "match_id"})
    away_feat = feat[feat["venue"] == "A"].drop(columns=["venue", "team"]).add_prefix("away_")
    away_feat = away_feat.rename(columns={"away_match_id": "match_id"})

    merged = base.merge(home_feat, on="match_id", how="left").merge(away_feat, on="match_id", how="left")

    # Pre-match xG (already in df)
    merged["pm_xg_diff"] = merged["pm_home_xg"].astype(float) - merged["pm_away_xg"].astype(float)
    merged["pm_xg_total"] = merged["pm_home_xg"].astype(float) + merged["pm_away_xg"].astype(float)
    return merged


def feature_columns(windows: tuple[int, ...] = (5, 10)) -> list[str]:
    cols: list[str] = []
    for side in ("home", "away"):
        for w in windows:
            for stat in ["corners_for", "corners_against", "shots_for", "shots_against", "sot_for", "sot_against"]:
                cols.append(f"{side}_{stat}_r{w}")
    cols += ["pm_home_xg", "pm_away_xg", "pm_xg_diff", "pm_xg_total"]
    return cols


def impute_league_defaults(X: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Fill early-season NaNs with sensible league defaults."""
    X = X.copy()
    for side in ("home", "away"):
        for w in windows:
            for stat, default in _LEAGUE_DEFAULTS.items():
                col = f"{side}_{stat}_r{w}"
                if col in X.columns:
                    X[col] = X[col].fillna(default)
    for c in ["pm_home_xg", "pm_away_xg", "pm_xg_diff", "pm_xg_total"]:
        if c in X.columns:
            X[c] = X[c].fillna(0.0).replace(0.0, np.nan).fillna(X[c].median() if X[c].notna().any() else 1.0)
    return X
