import numpy as np

from cooltwin import RoomParams, calibrate, compare_retrofits, fit_metrics, simulate, synthetic_hot_spell


def make_log(noise=0.2, seed=0):
    weather = synthetic_hot_spell(days=5, seed=5)
    truth = RoomParams(R_ie=0.04, R_roof=0.05, C_m=1.5e7, alpha_roof=0.7, g_win=0.4)
    t_in = simulate(weather, truth)["t_in"]
    log = weather.copy()
    log["t_in"] = t_in + np.random.default_rng(seed).normal(0, noise, len(t_in))
    return log, truth


def test_calibration_reaches_noise_floor():
    log, _ = make_log(noise=0.2)
    result = calibrate(log)
    assert result.metrics["rmse_K"] < 0.3
    assert abs(result.metrics["nmbe_pct"]) < 1.0


def test_fit_metrics_perfect_match():
    m = np.array([30.0, 31.0, 32.0])
    stats = fit_metrics(m, m)
    assert stats["rmse_K"] == 0.0
    assert stats["nmbe_pct"] == 0.0


def test_retrofit_table_shape_and_baseline():
    log, truth = make_log()
    table = compare_retrofits(log[["t_out", "solar"]], truth)
    assert table.index[0] == "baseline"
    assert table.loc["baseline", "degree_hours_saved"] == 0.0
    assert (table.drop(index="baseline")["degree_hours_saved"] > 0).all()
