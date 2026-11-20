"""Calibrate room parameters against measured indoor temperature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import RoomParams, simulate

# Parameters that are usually uncertain in an existing building, with
# physically plausible bounds. Calibration happens in log space so that
# parameters spanning orders of magnitude are treated evenly.
DEFAULT_BOUNDS: Dict[str, tuple] = {
    "R_ie": (0.005, 0.2),
    "R_im": (0.0005, 0.02),
    "R_roof": (0.005, 0.2),
    "C_i": (5.0e4, 2.0e6),
    "C_m": (1.0e6, 5.0e7),
    "alpha_roof": (0.2, 0.95),
    "g_win": (0.05, 3.0),
}


@dataclass
class CalibrationResult:
    params: RoomParams
    fitted: Dict[str, float]
    metrics: Dict[str, float]
    simulated: pd.Series


def fit_metrics(measured: np.ndarray, simulated: np.ndarray, n_params: int = 0) -> Dict[str, float]:
    """Goodness-of-fit statistics for a calibrated model.

    Returns RMSE and MAE in kelvin, plus CV(RMSE) and NMBE in percent,
    computed the way ASHRAE Guideline 14 defines them.
    """
    m = np.asarray(measured, dtype=float)
    s = np.asarray(simulated, dtype=float)
    resid = m - s
    n = len(m)
    dof = max(n - n_params, 1)
    mean = m.mean()
    rmse = float(np.sqrt(np.sum(resid**2) / dof))
    return {
        "rmse_K": rmse,
        "mae_K": float(np.mean(np.abs(resid))),
        "cv_rmse_pct": float(100.0 * rmse / mean),
        "nmbe_pct": float(100.0 * resid.sum() / (dof * mean)),
        "peak_error_K": float(m.max() - s.max()),
    }


def calibrate(
    data: pd.DataFrame,
    base: RoomParams | None = None,
    free: Sequence[str] = ("R_ie", "R_im", "R_roof", "C_i", "C_m", "alpha_roof", "g_win"),
    warmup_hours: float = 24.0,
) -> CalibrationResult:
    """Fit the free parameters so simulated ``t_in`` matches measured ``t_in``.

    The first ``warmup_hours`` are simulated but excluded from the error, so
    that the arbitrary initial state of the thermal mass does not bias the fit.
    """
    base = base or RoomParams()
    free = list(free)
    for name in free:
        if name not in DEFAULT_BOUNDS:
            raise KeyError(f"no bounds defined for parameter '{name}'")

    measured = data["t_in"].to_numpy(dtype=float)
    weather = data[["t_out", "solar"]]
    mask = data.index >= data.index[0] + pd.Timedelta(hours=warmup_hours)
    if mask.sum() < 10:
        raise ValueError("not enough data after the warm-up period")

    lo = np.log([DEFAULT_BOUNDS[n][0] for n in free])
    hi = np.log([DEFAULT_BOUNDS[n][1] for n in free])
    x0 = np.clip(np.log([getattr(base, n) for n in free]), lo + 1e-6, hi - 1e-6)

    def to_params(x: np.ndarray) -> RoomParams:
        return base.with_updates(**{n: float(np.exp(v)) for n, v in zip(free, x)})

    def residuals(x: np.ndarray) -> np.ndarray:
        sim = simulate(weather, to_params(x), t_init=measured[0])["t_in"].to_numpy()
        return (sim - measured)[mask]

    sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=200)
    params = to_params(sol.x)
    sim = simulate(weather, params, t_init=measured[0])["t_in"]
    metrics = fit_metrics(measured[mask], sim.to_numpy()[mask], n_params=len(free))
    fitted = {n: getattr(params, n) for n in free}
    return CalibrationResult(params=params, fitted=fitted, metrics=metrics, simulated=sim)
