"""Dixon-Coles bivariate Poisson model with time decay.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass
class DixonColesParams:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    ref_date: pd.Timestamp
    xi: float
    league_mean_goals: float = field(default=1.4)

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for (home, away). Unknown team -> league mean attack/defense."""
        a_h = self.attack.get(home, 0.0)
        d_h = self.defense.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_a = self.defense.get(away, 0.0)
        lam_h = np.exp(a_h + d_a + self.home_adv)
        lam_a = np.exp(a_a + d_h)
        return float(lam_h), float(lam_a)


# --------------------------------------------------------------------------------------
# Dixon-Coles tau adjustment for low scores
# --------------------------------------------------------------------------------------
def _tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _log_poisson(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """log P(K=k | Poisson(lam)). lam may include zeros — clipped for stability."""
    lam = np.clip(lam, 1e-9, None)
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _nll(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> float:
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[-2]
    rho = params[-1]

    log_lam_h = attack[home_idx] + defense[away_idx] + home_adv
    log_lam_a = attack[away_idx] + defense[home_idx]
    lam_h = np.exp(log_lam_h)
    lam_a = np.exp(log_lam_a)

    ll = _log_poisson(home_goals, lam_h) + _log_poisson(away_goals, lam_a)

    # Dixon-Coles tau correction (vectorised over low-score cells)
    tau = np.ones_like(lam_h)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau = np.where(m00, 1.0 - lam_h * lam_a * rho, tau)
    tau = np.where(m01, 1.0 + lam_h * rho, tau)
    tau = np.where(m10, 1.0 + lam_a * rho, tau)
    tau = np.where(m11, 1.0 - rho, tau)
    tau = np.clip(tau, 1e-9, None)
    ll = ll + np.log(tau)

    # Sum-zero constraints implemented as soft penalties (cheap and stable)
    penalty = 1000.0 * (attack.sum() ** 2 + defense.sum() ** 2)
    return -float(np.sum(weights * ll)) + penalty


def fit(
    df: pd.DataFrame,
    *,
    xi: float = 0.0025,
    ref_date: pd.Timestamp | None = None,
    target: str = "ft",
) -> DixonColesParams:
    """Fit Dixon-Coles model to completed matches in df.

    Required columns: date, home, away, home_goals, away_goals (or ht_home/ht_away when target='ht').
    """
    if target == "ft":
        gh = df["home_goals"].astype(int).values
        ga = df["away_goals"].astype(int).values
    elif target == "ht":
        gh = df["ht_home"].astype(int).values
        ga = df["ht_away"].astype(int).values
    else:
        raise ValueError(f"unknown target {target!r}")

    teams = sorted(set(df["home"]).union(set(df["away"])))
    team_idx = {t: i for i, t in enumerate(teams)}
    home_idx = df["home"].map(team_idx).values
    away_idx = df["away"].map(team_idx).values

    if ref_date is None:
        ref_date = pd.to_datetime(df["date"].max())
    days_back = (pd.to_datetime(ref_date) - pd.to_datetime(df["date"])).dt.days.values.astype(float)
    weights = np.exp(-xi * np.clip(days_back, 0, None))

    n = len(teams)
    league_mean = float(np.average(gh + ga, weights=weights)) if weights.sum() else float((gh + ga).mean())

    # Initial guesses
    x0 = np.zeros(2 * n + 2)
    x0[-2] = 0.25  # home advantage (log scale)
    x0[-1] = -0.05  # rho

    bounds = [(-3.0, 3.0)] * (2 * n) + [(-0.5, 1.5), (-0.25, 0.25)]

    res = minimize(
        _nll,
        x0,
        args=(n, home_idx, away_idx, gh, ga, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    params = res.x
    attack = {t: float(params[i]) for i, t in enumerate(teams)}
    defense = {t: float(params[n + i]) for i, t in enumerate(teams)}
    home_adv = float(params[-2])
    rho = float(params[-1])

    return DixonColesParams(
        teams=teams,
        attack=attack,
        defense=defense,
        home_adv=home_adv,
        rho=rho,
        ref_date=pd.to_datetime(ref_date),
        xi=xi,
        league_mean_goals=league_mean,
    )


def score_matrix(
    lam_h: float,
    lam_a: float,
    rho: float,
    *,
    max_goals: int = 10,
) -> np.ndarray:
    """Joint score probability matrix P[i, j] with Dixon-Coles correction on the four low-score cells."""
    i = np.arange(max_goals + 1)
    ph = np.exp(_log_poisson(i.astype(float), np.full_like(i, lam_h, dtype=float)))
    pa = np.exp(_log_poisson(i.astype(float), np.full_like(i, lam_a, dtype=float)))
    mat = np.outer(ph, pa)
    # Tau adjustment on the four low-score cells
    mat[0, 0] *= 1.0 - lam_h * lam_a * rho
    mat[0, 1] *= 1.0 + lam_h * rho
    mat[1, 0] *= 1.0 + lam_a * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0.0, None)
    s = mat.sum()
    if s > 0:
        mat = mat / s
    return mat


def predict_score_matrix(
    params: DixonColesParams,
    home: str,
    away: str,
    *,
    max_goals: int = 10,
) -> tuple[np.ndarray, float, float]:
    lam_h, lam_a = params.lambdas(home, away)
    mat = score_matrix(lam_h, lam_a, params.rho, max_goals=max_goals)
    return mat, lam_h, lam_a
