"""Load and normalize Peru Primera Division match data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_goal_minutes(raw: object) -> list[int]:
    """Parse a goal timings cell like '45,66+2,81' or "45'2,90'7" into minute ints.

    Stoppage time variants ("45+2", "45'2") collapse to the base minute.
    Empty/NaN/non-string returns an empty list.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip().strip('"')
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # Normalize "45+2" and "45'2" -> "45"
        tok = tok.split("+", 1)[0].split("'", 1)[0]
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


def _first_to_score(home_min: list[int], away_min: list[int]) -> str:
    """Return 'H', 'A', or 'N' (no goals)."""
    if not home_min and not away_min:
        return "N"
    h = min(home_min) if home_min else 10_000
    a = min(away_min) if away_min else 10_000
    if h < a:
        return "H"
    if a < h:
        return "A"
    # Same minute, very rare — give it to home arbitrarily; flag separately if needed
    return "H"


def _ht_goals(minutes: list[int]) -> int:
    return sum(1 for m in minutes if m <= 45)


def _load_one(path: str | Path, season: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["season"] = season
    return df


def load_seasons(paths: Iterable[tuple[str | Path, int]]) -> pd.DataFrame:
    """Load multiple season CSVs, normalize, and derive helper columns.

    `paths` is an iterable of (path, season_year) pairs.
    """
    frames = [_load_one(p, s) for p, s in paths]
    raw = pd.concat(frames, ignore_index=True)

    raw["date"] = pd.to_datetime(raw["date_GMT"], format="%b %d %Y - %I:%M%p", errors="coerce")
    # Fallback parser for any leftovers
    mask = raw["date"].isna()
    if mask.any():
        raw.loc[mask, "date"] = pd.to_datetime(raw.loc[mask, "date_GMT"], errors="coerce")

    raw["home_goal_minutes"] = raw["home_team_goal_timings"].apply(parse_goal_minutes)
    raw["away_goal_minutes"] = raw["away_team_goal_timings"].apply(parse_goal_minutes)
    raw["first_to_score"] = [
        _first_to_score(h, a)
        for h, a in zip(raw["home_goal_minutes"], raw["away_goal_minutes"])
    ]
    raw["ht_home_parsed"] = raw["home_goal_minutes"].apply(_ht_goals)
    raw["ht_away_parsed"] = raw["away_goal_minutes"].apply(_ht_goals)

    out = pd.DataFrame(
        {
            "date": raw["date"],
            "season": raw["season"],
            "gw": raw["Game Week"],
            "status": raw["status"],
            "home": raw["home_team_name"].astype(str).str.strip(),
            "away": raw["away_team_name"].astype(str).str.strip(),
            "referee": raw["referee"].astype(str).str.strip(),
            "home_goals": raw["home_team_goal_count"],
            "away_goals": raw["away_team_goal_count"],
            "ht_home": raw["home_team_goal_count_half_time"].fillna(raw["ht_home_parsed"]),
            "ht_away": raw["away_team_goal_count_half_time"].fillna(raw["ht_away_parsed"]),
            "home_goal_minutes": raw["home_goal_minutes"],
            "away_goal_minutes": raw["away_goal_minutes"],
            "first_to_score": raw["first_to_score"],
            "home_corners": raw["home_team_corner_count"],
            "away_corners": raw["away_team_corner_count"],
            "home_shots": raw["home_team_shots"],
            "away_shots": raw["away_team_shots"],
            "home_shots_on_target": raw["home_team_shots_on_target"],
            "away_shots_on_target": raw["away_team_shots_on_target"],
            "home_possession": raw["home_team_possession"],
            "away_possession": raw["away_team_possession"],
            "home_xg": raw["team_a_xg"],
            "away_xg": raw["team_b_xg"],
            "pm_home_xg": raw["Home Team Pre-Match xG"],
            "pm_away_xg": raw["Away Team Pre-Match xG"],
            "pm_home_ppg": raw["Pre-Match PPG (Home)"],
            "pm_away_ppg": raw["Pre-Match PPG (Away)"],
            "odds_h": raw["odds_ft_home_team_win"],
            "odds_d": raw["odds_ft_draw"],
            "odds_a": raw["odds_ft_away_team_win"],
            "odds_o15": raw["odds_ft_over15"],
            "odds_o25": raw["odds_ft_over25"],
            "odds_o35": raw["odds_ft_over35"],
            "odds_o45": raw["odds_ft_over45"],
            "odds_btts_y": raw["odds_btts_yes"],
            "odds_btts_n": raw["odds_btts_no"],
        }
    )

    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    return out


def split_completed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (completed, upcoming) frames."""
    completed = df[df["status"].str.lower() == "complete"].copy()
    upcoming = df[df["status"].str.lower() != "complete"].copy()
    return completed, upcoming


@dataclass
class DataPaths:
    """Default location of season CSVs inside this repo."""

    root: Path

    @classmethod
    def default(cls) -> "DataPaths":
        return cls(Path(__file__).resolve().parents[1] / "data")

    def files(self) -> list[tuple[Path, int]]:
        return [
            (self.root / "peru_2024.csv", 2024),
            (self.root / "peru_2025.csv", 2025),
            (self.root / "peru_2026.csv", 2026),
        ]


def load_default() -> pd.DataFrame:
    """Convenience wrapper for the bundled CSVs."""
    return load_seasons(DataPaths.default().files())
