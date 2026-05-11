"""Walk-forward backtest of the goal-markets model against closing odds.

For each completed match in the backtest window:
  1. Fit DC (FT + HT) using only matches strictly before its date.
  2. Predict markets.
  3. Score predictions (log-loss, Brier) and simulate Kelly + flat staking.

Refits are batched by week to keep the runtime tractable (~30s for one season).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dixon_coles import fit, DixonColesParams
from .markets import predict_markets, MarketPrediction
from .value import edge, kelly_fraction


def _won_1x2(row: pd.Series, selection: str) -> bool:
    h, a = int(row["home_goals"]), int(row["away_goals"])
    if selection == "Home":
        return h > a
    if selection == "Away":
        return a > h
    return h == a  # Draw


def _won_over(row: pd.Series, line: float) -> bool:
    return (int(row["home_goals"]) + int(row["away_goals"])) > line


def _won_btts(row: pd.Series, yes: bool) -> bool:
    btts_yes = int(row["home_goals"]) > 0 and int(row["away_goals"]) > 0
    return btts_yes if yes else not btts_yes


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    summary: pd.DataFrame
    metrics: dict
    bankroll: pd.DataFrame


def _safe_logloss(p: float, outcome: int) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return -(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))


def _market_outcomes_and_odds(row: pd.Series, pred: MarketPrediction) -> list[dict]:
    """Return one record per (market,selection) for this match with prob, odds, won flag."""
    rec: list[dict] = []
    for sel, p, odds_col in [
        ("Home", pred.p_home, "odds_h"),
        ("Draw", pred.p_draw, "odds_d"),
        ("Away", pred.p_away, "odds_a"),
    ]:
        rec.append(
            {
                "market": "1X2",
                "selection": sel,
                "p": p,
                "odds": float(row[odds_col]) if pd.notna(row[odds_col]) else np.nan,
                "won": _won_1x2(row, sel),
            }
        )
    for line, over_col in [(1.5, "odds_o15"), (2.5, "odds_o25"), (3.5, "odds_o35"), (4.5, "odds_o45")]:
        rec.append(
            {
                "market": f"Over {line}",
                "selection": f"Over {line}",
                "p": pred.p_over[line],
                "odds": float(row[over_col]) if pd.notna(row[over_col]) else np.nan,
                "won": _won_over(row, line),
            }
        )
    rec.append(
        {
            "market": "BTTS",
            "selection": "Yes",
            "p": pred.p_btts_yes,
            "odds": float(row["odds_btts_y"]) if pd.notna(row["odds_btts_y"]) else np.nan,
            "won": _won_btts(row, True),
        }
    )
    rec.append(
        {
            "market": "BTTS",
            "selection": "No",
            "p": pred.p_btts_no,
            "odds": float(row["odds_btts_n"]) if pd.notna(row["odds_btts_n"]) else np.nan,
            "won": _won_btts(row, False),
        }
    )
    return rec


def walk_forward(
    completed: pd.DataFrame,
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    xi: float = 0.0025,
    edge_threshold: float = 0.03,
    kelly_cap: float = 0.25,
    flat_stake: float = 1.0,
    initial_bankroll: float = 100.0,
    refit_freq_days: int = 7,
    verbose: bool = True,
) -> BacktestResult:
    """Run walk-forward backtest. Refits every `refit_freq_days` calendar days."""
    df = completed.copy().sort_values("date", kind="stable").reset_index(drop=True)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end) if end is not None else df["date"].max()

    window = df[(df["date"] >= start) & (df["date"] <= end)]
    if window.empty:
        raise ValueError("No matches found in backtest window.")

    records: list[dict] = []
    bankroll_kelly = initial_bankroll
    bankroll_flat = initial_bankroll
    bankroll_trace: list[dict] = []

    # Group target matches by refit week
    window = window.assign(_week=((window["date"] - start).dt.days // refit_freq_days).astype(int))
    last_week = -1
    ft_params: DixonColesParams | None = None
    ht_params: DixonColesParams | None = None

    iterator = window.groupby("_week", sort=True)
    if verbose:
        iterator = tqdm(iterator, desc="walk-forward")

    for week, group in iterator:
        cutoff = group["date"].min()
        train = df[df["date"] < cutoff]
        if len(train) < 40:  # need a minimum sample before fitting
            continue
        ft_params = fit(train, xi=xi, ref_date=cutoff, target="ft")
        ht_params = fit(train, xi=xi, ref_date=cutoff, target="ht")
        last_week = week

        for _, row in group.iterrows():
            pred = predict_markets(ft_params, ht_params, row["home"], row["away"])
            for rec in _market_outcomes_and_odds(row, pred):
                rec["date"] = row["date"]
                rec["home"] = row["home"]
                rec["away"] = row["away"]
                # Bet sizing
                p = rec["p"]
                odds = rec["odds"]
                if np.isnan(odds):
                    rec["bet_flat"] = 0.0
                    rec["bet_kelly"] = 0.0
                    rec["pnl_flat"] = 0.0
                    rec["pnl_kelly"] = 0.0
                    rec["edge"] = np.nan
                else:
                    e = edge(p, odds)
                    rec["edge"] = e
                    f_full, kstake = kelly_fraction(p, odds, fraction=kelly_cap)
                    bet = (e >= edge_threshold)
                    rec["bet_flat"] = flat_stake if bet else 0.0
                    rec["bet_kelly"] = kstake * bankroll_kelly if bet else 0.0
                    if bet:
                        if rec["won"]:
                            rec["pnl_flat"] = flat_stake * (odds - 1.0)
                            rec["pnl_kelly"] = rec["bet_kelly"] * (odds - 1.0)
                        else:
                            rec["pnl_flat"] = -flat_stake
                            rec["pnl_kelly"] = -rec["bet_kelly"]
                        bankroll_flat += rec["pnl_flat"]
                        bankroll_kelly += rec["pnl_kelly"]
                    else:
                        rec["pnl_flat"] = 0.0
                        rec["pnl_kelly"] = 0.0
                records.append(rec)

            bankroll_trace.append(
                {
                    "date": row["date"],
                    "bankroll_flat": bankroll_flat,
                    "bankroll_kelly": bankroll_kelly,
                }
            )

    bets = pd.DataFrame(records)
    # Per-market summary
    summary = _summarize(bets)
    # Overall metrics: log-loss and Brier per market group (1X2 family vs O/U vs BTTS)
    metrics = _metrics(bets)
    bankroll_df = pd.DataFrame(bankroll_trace)
    return BacktestResult(bets=bets, summary=summary, metrics=metrics, bankroll=bankroll_df)


def _summarize(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for market, sub in bets.groupby("market"):
        placed = sub[sub["bet_flat"] > 0]
        n_placed = len(placed)
        wins = placed["won"].sum()
        roi_flat = placed["pnl_flat"].sum() / max(placed["bet_flat"].sum(), 1e-9) if n_placed else 0.0
        kelly_stake_sum = placed["bet_kelly"].sum()
        roi_kelly = placed["pnl_kelly"].sum() / max(kelly_stake_sum, 1e-9) if kelly_stake_sum > 0 else 0.0
        rows.append(
            {
                "market": market,
                "n_observed": len(sub),
                "n_placed": n_placed,
                "hit_rate": wins / n_placed if n_placed else np.nan,
                "roi_flat": roi_flat,
                "pnl_flat": placed["pnl_flat"].sum(),
                "roi_kelly": roi_kelly,
                "pnl_kelly": placed["pnl_kelly"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("market").reset_index(drop=True)


def _metrics(bets: pd.DataFrame) -> dict:
    metrics: dict = {}
    if bets.empty:
        return metrics
    # 1X2 multi-class: pick rows where selection matches the realized outcome per match
    onex2 = bets[bets["market"] == "1X2"].copy()
    if not onex2.empty:
        per_match = onex2.pivot_table(
            index=["date", "home", "away"], columns="selection", values=["p", "won"], aggfunc="first"
        )
        # Brier (multi-class): sum_k (p_k - y_k)^2
        p_h = per_match[("p", "Home")].values
        p_d = per_match[("p", "Draw")].values
        p_a = per_match[("p", "Away")].values
        y_h = per_match[("won", "Home")].astype(int).values
        y_d = per_match[("won", "Draw")].astype(int).values
        y_a = per_match[("won", "Away")].astype(int).values
        brier = float(np.mean((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2))
        # Log loss: pick prob of realized outcome
        p_outcome = np.where(y_h == 1, p_h, np.where(y_d == 1, p_d, p_a))
        log_loss = float(-np.mean(np.log(np.clip(p_outcome, 1e-6, 1 - 1e-6))))
        metrics["1x2_brier"] = brier
        metrics["1x2_log_loss"] = log_loss
        metrics["1x2_n"] = int(len(per_match))

    for binary_market in [
        ("BTTS", "Yes"),
        ("Over 2.5", "Over 2.5"),
    ]:
        m, sel = binary_market
        sub = bets[(bets["market"] == m) & (bets["selection"] == sel)]
        if not sub.empty:
            p = sub["p"].values
            y = sub["won"].astype(int).values
            metrics[f"{m}_brier"] = float(np.mean((p - y) ** 2))
            metrics[f"{m}_log_loss"] = float(np.mean([_safe_logloss(pi, yi) for pi, yi in zip(p, y)]))
            metrics[f"{m}_n"] = int(len(sub))

    return metrics


def tune_xi(
    completed: pd.DataFrame,
    *,
    xis: Iterable[float] = (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075),
    holdout_start: pd.Timestamp | str = "2025-08-01",
    holdout_end: pd.Timestamp | str = "2025-12-31",
) -> pd.DataFrame:
    """Grid search xi by log-loss on the 1X2 market in a holdout window.

    No betting — pure probabilistic evaluation. Fast: trains once per xi on data
    before holdout_start (refits inside walk_forward would dominate runtime).
    """
    df = completed.copy().sort_values("date").reset_index(drop=True)
    holdout_start = pd.to_datetime(holdout_start)
    holdout_end = pd.to_datetime(holdout_end)
    train = df[df["date"] < holdout_start]
    holdout = df[(df["date"] >= holdout_start) & (df["date"] <= holdout_end)]

    rows = []
    for xi in xis:
        ft = fit(train, xi=xi, ref_date=holdout_start, target="ft")
        ht = fit(train, xi=xi, ref_date=holdout_start, target="ht")
        ll = 0.0
        n = 0
        for _, r in holdout.iterrows():
            pred = predict_markets(ft, ht, r["home"], r["away"])
            h, a = int(r["home_goals"]), int(r["away_goals"])
            p = pred.p_home if h > a else (pred.p_draw if h == a else pred.p_away)
            ll += -np.log(max(p, 1e-6))
            n += 1
        rows.append({"xi": xi, "log_loss_1x2": ll / max(n, 1), "n": n})
    return pd.DataFrame(rows)
