"""Value bet detection and Kelly staking."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueBet:
    market: str
    selection: str
    model_prob: float
    odds: float
    edge: float           # model_prob * odds - 1
    kelly_full: float     # full-Kelly fraction of bankroll
    kelly_stake: float    # after fractional-Kelly cap


def edge(model_prob: float, odds: float) -> float:
    """Expected return per unit stake: p*odds - 1."""
    if odds is None or odds <= 1.0:
        return float("-inf")
    return float(model_prob) * float(odds) - 1.0


def kelly_fraction(model_prob: float, odds: float, fraction: float = 0.25) -> tuple[float, float]:
    """Return (full_kelly, capped_kelly_stake) for decimal `odds`.

    f* = (b*p - q) / b where b = odds - 1.
    `fraction` is the cap multiplier on full Kelly (e.g. 0.25 = quarter Kelly).
    Negative full Kelly => no bet (return 0).
    """
    if odds is None or odds <= 1.0:
        return 0.0, 0.0
    b = float(odds) - 1.0
    p = float(model_prob)
    q = 1.0 - p
    f = (b * p - q) / b
    if f <= 0:
        return f, 0.0
    return f, max(0.0, min(1.0, f * fraction))


def find_value_bets(
    candidates: list[tuple[str, str, float, float]],
    *,
    threshold: float = 0.03,
    kelly_cap: float = 0.25,
) -> list[ValueBet]:
    """Filter (market, selection, p, odds) tuples to those with edge >= threshold."""
    out: list[ValueBet] = []
    for market, sel, p, o in candidates:
        if o is None or o <= 1.0 or p <= 0.0:
            continue
        e = edge(p, o)
        if e < threshold:
            continue
        f_full, stake = kelly_fraction(p, o, fraction=kelly_cap)
        out.append(
            ValueBet(
                market=market,
                selection=sel,
                model_prob=p,
                odds=o,
                edge=e,
                kelly_full=f_full,
                kelly_stake=stake,
            )
        )
    return out


def market_candidates_from_prediction(pred, odds_row: dict) -> list[tuple[str, str, float, float]]:
    """Build (market, selection, model_prob, odds) candidates from a MarketPrediction and an odds dict."""
    pairs: list[tuple[str, str, float, float]] = []
    pairs.append(("1X2", "Home", pred.p_home, _f(odds_row.get("odds_h"))))
    pairs.append(("1X2", "Draw", pred.p_draw, _f(odds_row.get("odds_d"))))
    pairs.append(("1X2", "Away", pred.p_away, _f(odds_row.get("odds_a"))))
    pairs.append(("BTTS", "Yes", pred.p_btts_yes, _f(odds_row.get("odds_btts_y"))))
    pairs.append(("BTTS", "No", pred.p_btts_no, _f(odds_row.get("odds_btts_n"))))
    line_to_odds = {
        1.5: ("odds_o15", None),
        2.5: ("odds_o25", None),
        3.5: ("odds_o35", None),
        4.5: ("odds_o45", None),
    }
    for line, (over_key, _) in line_to_odds.items():
        pairs.append((f"Over {line}", f"Over {line}", pred.p_over[line], _f(odds_row.get(over_key))))
    return pairs


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v <= 1.0:
        return None
    return v
