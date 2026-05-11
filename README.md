# peru-pred — Peru Primera Division Prediction Model

Hybrid soccer prediction model for the Peru Liga 1 (Primera Division). Built on
three seasons (2024 / 2025 / 2026-in-progress) of match-level data — ~760
completed matches with goal timings, shots, corners, possession, pre-match xG,
and closing odds.

## Stack

- **Dixon-Coles bivariate Poisson with time decay** for full-time goal markets
  (1X2, O/U, BTTS, correct score, first-to-score).
- **A second Dixon-Coles model fitted on first-half goals** (parsed from goal
  timings ≤ 45') for HT markets (HT 1X2, HT O/U 0.5/1.5, HT BTTS).
- **LightGBM Poisson regressors** for home and away corners; total over/under
  derived by simulation.
- **Value-bet detector + Kelly staking** versus CSV closing odds, with a
  walk-forward backtest.

## Install

```bash
pip install -e .
```

## Data

Three CSVs ship under `data/` and are loaded automatically:

- `peru_2024.csv` — 306 matches (complete)
- `peru_2025.csv` — 329 completed + 17 canceled
- `peru_2026.csv` — 124 completed + 29 upcoming

## CLI

```bash
# 1. Fit FT + HT + corners models and persist to models/
peru-pred train

# 2. Predict every market for a single fixture
peru-pred predict --home "Sporting Cristal" --away "Alianza Lima"

# 3. Predict every upcoming fixture (2026 in-progress season)
peru-pred upcoming --out predictions_upcoming.csv

# 4. Walk-forward backtest with Kelly staking
peru-pred backtest --start 2025-06-01 --edge 0.05 --kelly 0.25 --bankroll 100

# 5. List +EV bets among upcoming fixtures
peru-pred value --threshold 0.03

# Bonus: grid-search the time-decay xi
peru-pred tune-xi --start 2025-08-01 --end 2025-12-14
```

## Markets covered

| Group       | Markets                                                                                  |
|-------------|------------------------------------------------------------------------------------------|
| Full-time   | 1X2, Over/Under 1.5/2.5/3.5/4.5, BTTS yes/no, correct score (top 10), first to score     |
| Half-time   | HT 1X2, HT Over/Under 0.5/1.5, HT BTTS                                                   |
| Corners     | Expected home/away/total, Over/Under 7.5/8.5/9.5/10.5/11.5                               |

## Backtest output

`peru-pred backtest` writes three CSVs:

- `backtest_bets.csv` — every (match, market, selection) considered with model
  prob, odds, edge, flat & Kelly stakes, P&L, win/loss.
- `backtest_summary.csv` — per-market totals (placed, hit rate, ROI flat, ROI
  Kelly, P&L).
- `backtest_bankroll.csv` — bankroll trajectory for plotting.

Probabilistic metrics (log-loss, multi-class Brier) are printed in the
terminal.

## Package layout

```
peru_pred/
  data.py            # CSV loading + goal-timing parser
  features.py        # rolling per-team rates (causal)
  dixon_coles.py     # bivariate Poisson MLE + Dixon-Coles tau correction
  markets.py         # score matrix -> market probabilities
  corners.py         # LightGBM corners + Poisson simulation for totals
  value.py           # edge calc + Kelly staking
  backtest.py        # walk-forward evaluation + xi grid search
  cli.py             # `peru-pred ...` entry points
scripts/
  train_all.py       # one-shot fit + persist + backtest
tests/               # pytest unit tests
data/                # season CSVs
```

## Tuning notes

- **Time decay xi** defaults to 0.002 (≈ half-life of ~1 year). Lower values
  (0.001) gave slightly better 1X2 log-loss on the 2025 holdout; higher values
  (0.005) responded faster to recent form. Re-grid-search with `tune-xi`
  whenever fresh data is added.
- **Edge threshold** for the value bettor defaults to 5% (`--edge 0.05`).
  Lowering to 3% surfaces more bets but with lower confidence.
- **Quarter Kelly** is the default stake cap (`--kelly 0.25`). Use 0.10–0.25 for
  realism; full Kelly is heavily exposed to estimation error on a 760-match
  training set.

## Limitations

- The corners model has **no in-CSV odds**, so corners-market ROI is not
  computable. Predictions are evaluated by Brier and calibration only.
- Promoted teams (3 new entrants in 2026) start each season with priors at the
  league mean — expect wider error bars on their first ~5 matches.
- Game-week 1 of each season has zero pre-match xG / PPG in the source data;
  the loader handles these gracefully but those rows are noisier.
- The first-to-score market uses the closed-form race-to-first-event result for
  independent Poisson processes — it ignores the Dixon-Coles low-score tweak.
- **Bet responsibly.** All probabilities are model output, not advice.

## Tests

```bash
python3 -m pytest -q
```

21 tests covering goal-timing parsing, score-matrix normalization,
Dixon-Coles home-advantage recovery, market consistency, and Kelly math.
