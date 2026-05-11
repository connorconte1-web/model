"""Convert a Dixon-Coles score matrix into betting market probabilities.

Markets covered:
- 1X2 (home/draw/away)
- Over/Under N goals (any N)
- BTTS yes/no
- Correct score (top N)
- HT 1X2, HT over/under, HT BTTS (via a separate first-half score matrix)
- First to score (home / away / no goal) via simple two-half decomposition
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dixon_coles import DixonColesParams, score_matrix


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def result_1x2(mat: np.ndarray) -> tuple[float, float, float]:
    """(home, draw, away) probabilities from a score matrix."""
    home = float(np.tril(mat, -1).sum())
    draw = float(np.trace(mat))
    away = float(np.triu(mat, 1).sum())
    return home, draw, away


def total_goal_probs(mat: np.ndarray) -> np.ndarray:
    """Probability over total goals 0..(2*max_goals)."""
    n = mat.shape[0]
    totals = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            totals[i + j] += mat[i, j]
    return totals


def over_under(mat: np.ndarray, line: float) -> tuple[float, float]:
    """P(total > line), P(total < line). Half-goal lines only (no push)."""
    totals = total_goal_probs(mat)
    cutoff = int(np.floor(line)) + 1  # >= cutoff goals = OVER
    over = float(totals[cutoff:].sum())
    return over, 1.0 - over


def btts(mat: np.ndarray) -> tuple[float, float]:
    """(yes, no) — both teams to score."""
    p_home_zero = float(mat[0, :].sum())
    p_away_zero = float(mat[:, 0].sum())
    p_both_zero = float(mat[0, 0])
    no = p_home_zero + p_away_zero - p_both_zero
    yes = 1.0 - no
    return yes, no


def correct_score_top(mat: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    flat = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            flat.append((f"{i}-{j}", float(mat[i, j])))
    flat.sort(key=lambda t: -t[1])
    return flat[:k]


# --------------------------------------------------------------------------------------
# First-to-score
# --------------------------------------------------------------------------------------
def first_to_score(lam_h: float, lam_a: float) -> tuple[float, float, float]:
    """P(home scores first), P(away scores first), P(0-0).

    Standard result for independent Poisson processes — race-to-first event:
        P(no goals) = exp(-(lam_h + lam_a))
        P(home first) = (lam_h / (lam_h + lam_a)) * (1 - P(no goals))
    Note: ignores the Dixon-Coles low-score correction, but matches the
    underlying generating process closely enough for this market.
    """
    total = lam_h + lam_a
    if total <= 0:
        return 0.0, 0.0, 1.0
    no_goal = float(np.exp(-total))
    home_first = (lam_h / total) * (1.0 - no_goal)
    away_first = (lam_a / total) * (1.0 - no_goal)
    return float(home_first), float(away_first), no_goal


# --------------------------------------------------------------------------------------
# Full market bundle
# --------------------------------------------------------------------------------------
@dataclass
class MarketPrediction:
    home: str
    away: str
    lam_h: float
    lam_a: float
    lam_h_ht: float
    lam_a_ht: float
    # FT
    p_home: float
    p_draw: float
    p_away: float
    p_over: dict[float, float]  # keyed by line (1.5, 2.5, 3.5, 4.5)
    p_under: dict[float, float]
    p_btts_yes: float
    p_btts_no: float
    correct_score: list[tuple[str, float]]
    # First-to-score
    p_fts_home: float
    p_fts_away: float
    p_fts_none: float
    # HT
    p_ht_home: float
    p_ht_draw: float
    p_ht_away: float
    p_ht_over: dict[float, float]  # 0.5, 1.5
    p_ht_under: dict[float, float]
    p_ht_btts_yes: float
    p_ht_btts_no: float


_FT_LINES: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)
_HT_LINES: tuple[float, ...] = (0.5, 1.5)


def predict_markets(
    ft_params: DixonColesParams,
    ht_params: DixonColesParams,
    home: str,
    away: str,
    *,
    max_goals: int = 10,
    ft_lines: Iterable[float] = _FT_LINES,
    ht_lines: Iterable[float] = _HT_LINES,
) -> MarketPrediction:
    """Compute every supported market probability for one fixture."""
    lam_h, lam_a = ft_params.lambdas(home, away)
    mat_ft = score_matrix(lam_h, lam_a, ft_params.rho, max_goals=max_goals)
    ph, pd_, pa = result_1x2(mat_ft)
    over_ft: dict[float, float] = {}
    under_ft: dict[float, float] = {}
    for L in ft_lines:
        o, u = over_under(mat_ft, L)
        over_ft[L], under_ft[L] = o, u
    btts_y, btts_n = btts(mat_ft)
    cs = correct_score_top(mat_ft, k=10)
    fts_h, fts_a, fts_n = first_to_score(lam_h, lam_a)

    # First half (separate fit on HT goals)
    lh_ht, la_ht = ht_params.lambdas(home, away)
    mat_ht = score_matrix(lh_ht, la_ht, ht_params.rho, max_goals=max(6, max_goals // 2))
    ph_ht, pd_ht, pa_ht = result_1x2(mat_ht)
    over_ht: dict[float, float] = {}
    under_ht: dict[float, float] = {}
    for L in ht_lines:
        o, u = over_under(mat_ht, L)
        over_ht[L], under_ht[L] = o, u
    btts_y_ht, btts_n_ht = btts(mat_ht)

    return MarketPrediction(
        home=home,
        away=away,
        lam_h=lam_h,
        lam_a=lam_a,
        lam_h_ht=lh_ht,
        lam_a_ht=la_ht,
        p_home=ph,
        p_draw=pd_,
        p_away=pa,
        p_over=over_ft,
        p_under=under_ft,
        p_btts_yes=btts_y,
        p_btts_no=btts_n,
        correct_score=cs,
        p_fts_home=fts_h,
        p_fts_away=fts_a,
        p_fts_none=fts_n,
        p_ht_home=ph_ht,
        p_ht_draw=pd_ht,
        p_ht_away=pa_ht,
        p_ht_over=over_ht,
        p_ht_under=under_ht,
        p_ht_btts_yes=btts_y_ht,
        p_ht_btts_no=btts_n_ht,
    )


def market_to_dict(p: MarketPrediction) -> dict[str, float | str]:
    """Flatten a MarketPrediction into a single-row dict suitable for CSV output."""
    d: dict[str, float | str] = {
        "home": p.home,
        "away": p.away,
        "lam_h": p.lam_h,
        "lam_a": p.lam_a,
        "p_home": p.p_home,
        "p_draw": p.p_draw,
        "p_away": p.p_away,
        "p_btts_yes": p.p_btts_yes,
        "p_btts_no": p.p_btts_no,
        "p_fts_home": p.p_fts_home,
        "p_fts_away": p.p_fts_away,
        "p_fts_none": p.p_fts_none,
        "p_ht_home": p.p_ht_home,
        "p_ht_draw": p.p_ht_draw,
        "p_ht_away": p.p_ht_away,
        "p_ht_btts_yes": p.p_ht_btts_yes,
        "p_ht_btts_no": p.p_ht_btts_no,
    }
    for L, v in p.p_over.items():
        d[f"p_over_{L}"] = v
    for L, v in p.p_under.items():
        d[f"p_under_{L}"] = v
    for L, v in p.p_ht_over.items():
        d[f"p_ht_over_{L}"] = v
    for L, v in p.p_ht_under.items():
        d[f"p_ht_under_{L}"] = v
    for i, (score, prob) in enumerate(p.correct_score[:5]):
        d[f"cs_{i+1}"] = score
        d[f"cs_{i+1}_p"] = prob
    return d
