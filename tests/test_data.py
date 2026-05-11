import pandas as pd

from peru_pred.data import (
    _first_to_score,
    _ht_goals,
    load_default,
    parse_goal_minutes,
    split_completed,
)


def test_parse_goal_minutes_empty():
    assert parse_goal_minutes("") == []
    assert parse_goal_minutes(None) == []
    assert parse_goal_minutes(float("nan")) == []


def test_parse_goal_minutes_basic():
    assert parse_goal_minutes("45,66,81") == [45, 66, 81]


def test_parse_goal_minutes_stoppage_variants():
    # Both '90+2' and "90'7" should collapse to base minute
    assert parse_goal_minutes("45+2,90+7") == [45, 90]
    assert parse_goal_minutes("45'2,90'7") == [45, 90]


def test_parse_goal_minutes_mixed():
    assert parse_goal_minutes('"19,66+1"') == [19, 66]


def test_first_to_score():
    assert _first_to_score([], []) == "N"
    assert _first_to_score([10, 80], [70]) == "H"
    assert _first_to_score([70], [10, 80]) == "A"
    assert _first_to_score([], [70]) == "A"


def test_ht_goals_inclusive_45():
    assert _ht_goals([10, 45, 46]) == 2
    assert _ht_goals([90]) == 0
    assert _ht_goals([]) == 0


def test_load_default_shape_and_chronology():
    df = load_default()
    # Three seasons, ~800 rows
    assert 700 <= len(df) <= 900
    # Chronologically sorted
    assert df["date"].is_monotonic_increasing
    # HT parsed reconciles with stored half-time counts on ~all completed matches
    c, _ = split_completed(df)
    parsed_h = c["home_goal_minutes"].apply(_ht_goals)
    parsed_a = c["away_goal_minutes"].apply(_ht_goals)
    mismatch_h = (parsed_h != c["ht_home"].astype(int)).sum()
    mismatch_a = (parsed_a != c["ht_away"].astype(int)).sum()
    # Allow up to 1% drift in case of malformed timing rows
    assert mismatch_h <= max(3, len(c) // 100)
    assert mismatch_a <= max(3, len(c) // 100)


def test_split_completed_partition():
    df = load_default()
    c, u = split_completed(df)
    assert len(c) + len(u) == len(df)
    assert (c["status"].str.lower() == "complete").all()
