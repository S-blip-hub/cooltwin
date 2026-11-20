import numpy as np
import pandas as pd
import pytest

from cooltwin import RoomParams, simulate, synthetic_hot_spell
from cooltwin.model import ventilation_conductance


def constant_weather(t_out=30.0, solar=0.0, hours=72):
    idx = pd.date_range("2026-05-01", periods=hours * 6, freq="10min")
    return pd.DataFrame({"t_out": t_out, "solar": solar}, index=idx)


def test_no_gains_relaxes_to_outdoor():
    params = RoomParams(q_internal=0.0, g_win=0.0, alpha_roof=0.0)
    out = simulate(constant_weather(30.0, hours=240), params, t_init=36.0)
    assert out["t_in"].iloc[-1] == pytest.approx(30.0, abs=0.05)
    assert out["t_mass"].iloc[-1] == pytest.approx(30.0, abs=0.05)


def test_internal_gain_steady_state():
    # With only internal gain, steady-state rise equals Q / total conductance to outdoors.
    p = RoomParams(q_internal=200.0, g_win=0.0, alpha_roof=0.0, ach_day=2.0, ach_night=2.0)
    out = simulate(constant_weather(30.0, hours=480), p, t_init=30.0)
    g_mass_path = 1.0 / (p.R_im + p.R_roof)
    g_total = 1.0 / p.R_ie + ventilation_conductance(2.0, p.volume) + g_mass_path
    assert out["t_in"].iloc[-1] - 30.0 == pytest.approx(200.0 / g_total, rel=0.02)


def test_cool_roof_lowers_peak():
    w = synthetic_hot_spell(days=4)
    hot = simulate(w, RoomParams(alpha_roof=0.8))["t_in"]
    cool = simulate(w, RoomParams(alpha_roof=0.3))["t_in"]
    assert cool.max() < hot.max()


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        simulate(pd.DataFrame({"t_out": [30.0, 31.0]}), RoomParams())


def test_ventilation_conductance_value():
    # 36 m3 at 1 ACH -> 1.2 * 1005 * 36 / 3600 = 12.06 W/K
    assert ventilation_conductance(1.0, 36.0) == pytest.approx(12.06, rel=1e-3)
