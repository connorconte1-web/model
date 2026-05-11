"""Command-line interface for the Peru Primera prediction model."""
from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tabulate import tabulate

from .backtest import tune_xi, walk_forward
from .corners import fit_corners, predict_corners
from .data import DataPaths, load_default, split_completed
from .dixon_coles import fit
from .markets import market_to_dict, predict_markets
from .value import edge, find_value_bets, kelly_fraction, market_candidates_from_prediction


DEFAULT_MODELS_DIR = Path("models")


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_default()
    return split_completed(df)


@click.group(help="Peru Primera Division prediction model CLI.")
def main() -> None:
    pass


# -------- train ----------------------------------------------------------------------
@main.command(help="Fit FT/HT Dixon-Coles + corners models on all completed matches and persist.")
@click.option("--out", "out_dir", type=click.Path(), default=str(DEFAULT_MODELS_DIR), show_default=True)
@click.option("--xi", type=float, default=0.002, show_default=True, help="Time decay rate.")
def train(out_dir: str, xi: float) -> None:
    completed, _ = _load()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    click.echo(f"Fitting on {len(completed)} completed matches (xi={xi}) ...")
    ft = fit(completed, xi=xi, target="ft")
    ht = fit(completed, xi=xi, target="ht")
    corners = fit_corners(completed)
    with open(out / "ft.pkl", "wb") as fh:
        pickle.dump(ft, fh)
    with open(out / "ht.pkl", "wb") as fh:
        pickle.dump(ht, fh)
    with open(out / "corners.pkl", "wb") as fh:
        pickle.dump(corners, fh)
    click.echo(f"Saved models to {out}/")
    click.echo(f"  home_adv={ft.home_adv:.3f}  rho={ft.rho:.4f}  league_mean={ft.league_mean_goals:.2f}")


def _load_models(models_dir: Path) -> tuple:
    with open(models_dir / "ft.pkl", "rb") as fh:
        ft = pickle.load(fh)
    with open(models_dir / "ht.pkl", "rb") as fh:
        ht = pickle.load(fh)
    corners_path = models_dir / "corners.pkl"
    corners = None
    if corners_path.exists():
        with open(corners_path, "rb") as fh:
            corners = pickle.load(fh)
    return ft, ht, corners


# -------- predict --------------------------------------------------------------------
@main.command(help="Predict markets for a single fixture.")
@click.option("--home", required=True)
@click.option("--away", required=True)
@click.option("--models", "models_dir", type=click.Path(), default=str(DEFAULT_MODELS_DIR), show_default=True)
def predict(home: str, away: str, models_dir: str) -> None:
    ft, ht, corners_model = _load_models(Path(models_dir))
    pred = predict_markets(ft, ht, home, away)
    rows = [
        ["1X2 Home", f"{pred.p_home:.3f}"],
        ["1X2 Draw", f"{pred.p_draw:.3f}"],
        ["1X2 Away", f"{pred.p_away:.3f}"],
        ["lambda_h, lambda_a", f"{pred.lam_h:.2f}, {pred.lam_a:.2f}"],
        ["Over 1.5 / Under 1.5", f"{pred.p_over[1.5]:.3f} / {pred.p_under[1.5]:.3f}"],
        ["Over 2.5 / Under 2.5", f"{pred.p_over[2.5]:.3f} / {pred.p_under[2.5]:.3f}"],
        ["Over 3.5 / Under 3.5", f"{pred.p_over[3.5]:.3f} / {pred.p_under[3.5]:.3f}"],
        ["Over 4.5 / Under 4.5", f"{pred.p_over[4.5]:.3f} / {pred.p_under[4.5]:.3f}"],
        ["BTTS Yes / No", f"{pred.p_btts_yes:.3f} / {pred.p_btts_no:.3f}"],
        ["First to score: Home / Away / No goal", f"{pred.p_fts_home:.3f} / {pred.p_fts_away:.3f} / {pred.p_fts_none:.3f}"],
        ["HT 1X2 Home / Draw / Away", f"{pred.p_ht_home:.3f} / {pred.p_ht_draw:.3f} / {pred.p_ht_away:.3f}"],
        ["HT Over 0.5 / Under 0.5", f"{pred.p_ht_over[0.5]:.3f} / {pred.p_ht_under[0.5]:.3f}"],
        ["HT Over 1.5 / Under 1.5", f"{pred.p_ht_over[1.5]:.3f} / {pred.p_ht_under[1.5]:.3f}"],
        ["HT BTTS Yes / No", f"{pred.p_ht_btts_yes:.3f} / {pred.p_ht_btts_no:.3f}"],
    ]
    click.echo(f"\n{home} vs {away}")
    click.echo(tabulate(rows, headers=["Market", "Probability"], tablefmt="github"))
    click.echo("\nTop correct scores:")
    click.echo(tabulate(pred.correct_score[:8], headers=["Score", "Prob"], tablefmt="github", floatfmt=".3f"))

    if corners_model is not None:
        completed, _ = _load()
        synthetic = pd.DataFrame(
            {
                "date": [completed["date"].max() + pd.Timedelta(days=1)],
                "home": [home],
                "away": [away],
                "home_corners": [np.nan],
                "away_corners": [np.nan],
                "home_shots": [np.nan],
                "away_shots": [np.nan],
                "home_shots_on_target": [np.nan],
                "away_shots_on_target": [np.nan],
                "pm_home_xg": [np.nan],
                "pm_away_xg": [np.nan],
            }
        )
        corner_pred = predict_corners(corners_model, completed, synthetic, n_sim=20_000)
        cr = corner_pred.iloc[0]
        click.echo("\nCorners (LightGBM):")
        click.echo(
            tabulate(
                [
                    ["Expected home", f"{cr['lam_corners_home']:.2f}"],
                    ["Expected away", f"{cr['lam_corners_away']:.2f}"],
                    ["Expected total", f"{cr['lam_corners_total']:.2f}"],
                    ["P(over 8.5)", f"{cr['p_corners_over_8.5']:.3f}"],
                    ["P(over 9.5)", f"{cr['p_corners_over_9.5']:.3f}"],
                    ["P(over 10.5)", f"{cr['p_corners_over_10.5']:.3f}"],
                ],
                headers=["Market", "Value"],
                tablefmt="github",
            )
        )


# -------- upcoming -------------------------------------------------------------------
@main.command(help="Predict every upcoming fixture (status != complete) and write to CSV.")
@click.option("--out", "out_path", type=click.Path(), default="predictions_upcoming.csv", show_default=True)
@click.option("--models", "models_dir", type=click.Path(), default=str(DEFAULT_MODELS_DIR), show_default=True)
def upcoming(out_path: str, models_dir: str) -> None:
    ft, ht, corners_model = _load_models(Path(models_dir))
    completed, upcoming = _load()
    if upcoming.empty:
        click.echo("No upcoming fixtures.")
        return
    rows: list[dict] = []
    for _, r in upcoming.iterrows():
        pred = predict_markets(ft, ht, r["home"], r["away"])
        d = market_to_dict(pred)
        d["date"] = r["date"]
        d["gw"] = r["gw"]
        rows.append(d)
    out_df = pd.DataFrame(rows)

    if corners_model is not None and not upcoming.empty:
        corner_pred = predict_corners(corners_model, completed, upcoming, n_sim=20_000)
        corner_pred = corner_pred.reset_index(drop=True)
        out_df = pd.concat([out_df.reset_index(drop=True), corner_pred], axis=1)

    out_df.to_csv(out_path, index=False)
    click.echo(f"Wrote {len(out_df)} predictions -> {out_path}")


# -------- backtest -------------------------------------------------------------------
@main.command(help="Walk-forward backtest of goal markets with Kelly + flat staking.")
@click.option("--start", default="2025-06-01", show_default=True)
@click.option("--end", default=None)
@click.option("--xi", type=float, default=0.002, show_default=True)
@click.option("--edge", "edge_threshold", type=float, default=0.05, show_default=True, help="Min edge to place bet.")
@click.option("--kelly", "kelly_cap", type=float, default=0.25, show_default=True)
@click.option("--bankroll", type=float, default=100.0, show_default=True)
@click.option("--bets-out", type=click.Path(), default="backtest_bets.csv", show_default=True)
@click.option("--summary-out", type=click.Path(), default="backtest_summary.csv", show_default=True)
def backtest(start, end, xi, edge_threshold, kelly_cap, bankroll, bets_out, summary_out) -> None:
    completed, _ = _load()
    res = walk_forward(
        completed,
        start=start,
        end=end,
        xi=xi,
        edge_threshold=edge_threshold,
        kelly_cap=kelly_cap,
        initial_bankroll=bankroll,
    )
    click.echo("\n=== Per-market summary ===")
    click.echo(tabulate(res.summary, headers="keys", tablefmt="github", floatfmt=".3f", showindex=False))
    click.echo("\n=== Probabilistic metrics ===")
    for k, v in res.metrics.items():
        click.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    res.bets.to_csv(bets_out, index=False)
    res.summary.to_csv(summary_out, index=False)
    if not res.bankroll.empty:
        res.bankroll.to_csv("backtest_bankroll.csv", index=False)
        click.echo(f"\nFinal bankroll (flat={res.bankroll.iloc[-1]['bankroll_flat']:.2f}, "
                   f"kelly={res.bankroll.iloc[-1]['bankroll_kelly']:.2f})")
    click.echo(f"\nWrote bets -> {bets_out}, summary -> {summary_out}")


# -------- tune-xi --------------------------------------------------------------------
@main.command("tune-xi", help="Grid-search time-decay xi on a holdout window.")
@click.option("--start", default="2025-08-01", show_default=True)
@click.option("--end", default="2025-12-14", show_default=True)
def tune_xi_cmd(start: str, end: str) -> None:
    completed, _ = _load()
    res = tune_xi(completed, holdout_start=start, holdout_end=end)
    click.echo(tabulate(res, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))


# -------- value ----------------------------------------------------------------------
@main.command(help="List +EV bets among upcoming fixtures.")
@click.option("--threshold", type=float, default=0.03, show_default=True, help="Minimum edge.")
@click.option("--kelly", "kelly_cap", type=float, default=0.25, show_default=True)
@click.option("--models", "models_dir", type=click.Path(), default=str(DEFAULT_MODELS_DIR), show_default=True)
@click.option("--out", "out_path", type=click.Path(), default="value_bets.csv", show_default=True)
def value(threshold: float, kelly_cap: float, models_dir: str, out_path: str) -> None:
    ft, ht, _ = _load_models(Path(models_dir))
    _, upcoming = _load()
    if upcoming.empty:
        click.echo("No upcoming fixtures.")
        return
    rows: list[dict] = []
    for _, r in upcoming.iterrows():
        pred = predict_markets(ft, ht, r["home"], r["away"])
        candidates = market_candidates_from_prediction(pred, r.to_dict())
        for vb in find_value_bets(candidates, threshold=threshold, kelly_cap=kelly_cap):
            rows.append(
                {
                    "date": r["date"],
                    "home": r["home"],
                    "away": r["away"],
                    "market": vb.market,
                    "selection": vb.selection,
                    "model_prob": round(vb.model_prob, 4),
                    "odds": round(vb.odds, 3),
                    "edge": round(vb.edge, 4),
                    "kelly_full": round(vb.kelly_full, 4),
                    "kelly_stake_pct": round(vb.kelly_stake, 4),
                }
            )
    if not rows:
        click.echo(f"No value bets above edge {threshold:.0%}.")
        return
    out_df = pd.DataFrame(rows).sort_values("edge", ascending=False)
    click.echo(tabulate(out_df, headers="keys", tablefmt="github", showindex=False))
    out_df.to_csv(out_path, index=False)
    click.echo(f"\nWrote {len(out_df)} value bets -> {out_path}")


if __name__ == "__main__":
    main()
