"""Unit tests for shared feature engineering (power in kW, rule flags)."""
from __future__ import annotations

import numpy as np
import pytest

from feature_engineering import (
    PWF_KW_MAX,
    PWF_KW_MIN,
    build_feature_row,
    explanation_factors,
    power_kw,
    row_to_vector,
)


def test_power_kw_mid_range() -> None:
    p = power_kw(1500, 35)
    assert 5.0 < p < 6.5


def test_power_kw_low_triggers_pwf() -> None:
    row = build_feature_row(298.0, 308.0, 1000.0, 10.0, 40.0)
    assert row["power_kw"] < PWF_KW_MIN
    assert row["pwf_risk"] == 1


def test_power_kw_in_band_no_pwf() -> None:
    row = build_feature_row(298.0, 308.0, 1600.0, 32.0, 80.0)
    assert PWF_KW_MIN <= row["power_kw"] <= PWF_KW_MAX
    assert row["pwf_risk"] == 0


def test_hdf_risk_flag() -> None:
    row = build_feature_row(300.0, 308.0, 1200.0, 30.0, 50.0)
    assert row["temp_diff"] < 8.6
    assert row["hdf_risk"] == 1


def test_vector_order_matches_feature_list() -> None:
    row = build_feature_row(298, 308, 1500, 35, 100)
    v = row_to_vector(row)
    assert v.shape == (1, 12)
    assert not np.isnan(v).any()


def test_explanation_non_empty() -> None:
    row = build_feature_row(298, 308, 1500, 35, 100)
    xs = explanation_factors(row)
    assert isinstance(xs, list)
    assert len(xs) >= 1
