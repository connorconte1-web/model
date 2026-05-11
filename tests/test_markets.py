import numpy as np
import pytest

from peru_pred.data import load_default, split_completed
from peru_pred.dixon_coles import fit, score_matrix
from peru_pred.markets import (
    btts,
    first_to_score,
    over_under,
    predict_markets,
    result_1x2,
)
from peru_pred.value import edge, kelly_fraction


def test_1x2_sums_to_one():
    mat = score_matrix(1.4, 1.0, rho=-0.05, max_goals=10)
    h, d, a = result_1x2(mat)
    assert pytest.approx(h + d + a, abs=1e-9) == 1.0


def test_over_under_complements():
    mat = score_matrix(1.4, 1.0, rho=-0.05, max_goals=10)
    for line in (1.5, 2.5, 3.5):
        o, u = over_under(mat, line)
        assert pytest.approx(o + u, abs=1e-9) == 1.0


def test_btts_complements():
    mat = score_matrix(1.4, 1.0, rho=-0.05, max_goals=10)
    y, n = btts(mat)
    assert pytest.approx(y + n, abs=1e-9) == 1.0
    assert 0 <= y <= 1


def test_first_to_score_sums_to_one():
    h, a, n = first_to_score(1.2, 0.8)
    assert pytest.approx(h + a + n, abs=1e-9) == 1.0
    # Symmetry: equal lambdas -> equal home/away first
    h2, a2, _ = first_to_score(1.0, 1.0)
    assert abs(h2 - a2) < 1e-9


def test_first_to_score_zero_lambdas():
    h, a, n = first_to_score(0.0, 0.0)
    assert n == 1.0
    assert h == 0.0 and a == 0.0


def test_predict_markets_full_bundle():
    df = load_default()
    c, _ = split_completed(df)
    ft = fit(c.head(400), xi=0.002, target="ft")
    ht = fit(c.head(400), xi=0.002, target="ht")
    teams = list(ft.attack.keys())
    pred = predict_markets(ft, ht, teams[0], teams[1])
    assert pytest.approx(pred.p_home + pred.p_draw + pred.p_away, abs=1e-9) == 1.0
    assert pytest.approx(pred.p_btts_yes + pred.p_btts_no, abs=1e-9) == 1.0
    for L in (1.5, 2.5, 3.5, 4.5):
        assert pytest.approx(pred.p_over[L] + pred.p_under[L], abs=1e-9) == 1.0
    assert pytest.approx(pred.p_fts_home + pred.p_fts_away + pred.p_fts_none, abs=1e-9) == 1.0


def test_value_edge_and_kelly():
    # 60% true prob at odds 2.0 -> edge = 0.6 * 2.0 - 1 = 0.2
    assert pytest.approx(edge(0.6, 2.0), abs=1e-9) == 0.2
    full, capped = kelly_fraction(0.6, 2.0, fraction=0.5)
    # f* = (1*0.6 - 0.4)/1 = 0.2
    assert pytest.approx(full, abs=1e-9) == 0.2
    assert pytest.approx(capped, abs=1e-9) == 0.1


def test_value_negative_edge_no_bet():
    full, capped = kelly_fraction(0.4, 2.0, fraction=0.25)
    assert capped == 0.0
