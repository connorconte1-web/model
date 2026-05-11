import numpy as np
import pandas as pd
import pytest

from peru_pred.data import load_default, split_completed
from peru_pred.dixon_coles import fit, predict_score_matrix, score_matrix


def test_score_matrix_sums_to_one():
    mat = score_matrix(1.5, 1.1, rho=-0.05, max_goals=12)
    assert pytest.approx(mat.sum(), abs=1e-9) == 1.0
    assert (mat >= 0).all()


def test_score_matrix_rho_zero_factors():
    # With rho=0 the joint matrix should be the outer product of marginals
    mat = score_matrix(1.5, 1.1, rho=0.0, max_goals=12)
    margH = mat.sum(axis=1)
    margA = mat.sum(axis=0)
    expected = np.outer(margH, margA)
    assert np.allclose(mat, expected, atol=1e-9)


def test_fit_recovers_home_advantage():
    df = load_default()
    c, _ = split_completed(df)
    params = fit(c, xi=0.002)
    # Home advantage should be positive in any reasonable league sample
    assert params.home_adv > 0.1
    assert -0.2 < params.rho < 0.2
    # Sum-zero constraint approximately satisfied
    assert abs(sum(params.attack.values())) < 0.1
    assert abs(sum(params.defense.values())) < 0.1


def test_fit_predict_lambdas_positive():
    df = load_default()
    c, _ = split_completed(df)
    params = fit(c.head(300), xi=0.002)
    team_a, team_b = list(params.attack.keys())[:2]
    lh, la = params.lambdas(team_a, team_b)
    assert lh > 0 and la > 0


def test_predict_score_matrix_consistency():
    df = load_default()
    c, _ = split_completed(df)
    params = fit(c.head(400), xi=0.002)
    home, away = list(params.attack.keys())[:2]
    mat, lh, la = predict_score_matrix(params, home, away, max_goals=8)
    assert pytest.approx(mat.sum(), abs=1e-9) == 1.0
    # Marginal mean should be close to lambda (Dixon-Coles only nudges low cells)
    n = mat.shape[0]
    idx = np.arange(n)
    marg_h_mean = (mat.sum(axis=1) * idx).sum()
    marg_a_mean = (mat.sum(axis=0) * idx).sum()
    assert abs(marg_h_mean - lh) < 0.05
    assert abs(marg_a_mean - la) < 0.05
