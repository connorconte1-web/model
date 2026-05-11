"""End-to-end training + backtest helper.

Fits FT/HT Dixon-Coles and the corners model on all completed matches, runs a
walk-forward backtest from 2025-06-01 onward, and writes:
  - models/ft.pkl, models/ht.pkl, models/corners.pkl
  - backtest_bets.csv, backtest_summary.csv, backtest_bankroll.csv
"""
from __future__ import annotations

import pickle
from pathlib import Path

from peru_pred.backtest import walk_forward
from peru_pred.corners import fit_corners
from peru_pred.data import load_default, split_completed
from peru_pred.dixon_coles import fit


def main() -> None:
    completed, _ = split_completed(load_default())
    print(f"Loaded {len(completed)} completed matches.")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print("Fitting FT Dixon-Coles ...")
    ft = fit(completed, xi=0.002, target="ft")
    print("Fitting HT Dixon-Coles ...")
    ht = fit(completed, xi=0.002, target="ht")
    print("Fitting corners model ...")
    corners = fit_corners(completed)

    with open(models_dir / "ft.pkl", "wb") as fh:
        pickle.dump(ft, fh)
    with open(models_dir / "ht.pkl", "wb") as fh:
        pickle.dump(ht, fh)
    with open(models_dir / "corners.pkl", "wb") as fh:
        pickle.dump(corners, fh)
    print(f"Persisted models -> {models_dir}/")

    print("\nRunning walk-forward backtest from 2025-06-01 ...")
    res = walk_forward(
        completed,
        start="2025-06-01",
        xi=0.002,
        edge_threshold=0.05,
        kelly_cap=0.25,
        initial_bankroll=100.0,
    )
    res.bets.to_csv("backtest_bets.csv", index=False)
    res.summary.to_csv("backtest_summary.csv", index=False)
    if not res.bankroll.empty:
        res.bankroll.to_csv("backtest_bankroll.csv", index=False)
    print("\nPer-market summary:")
    print(res.summary.to_string(index=False))
    print("\nMetrics:")
    for k, v in res.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
