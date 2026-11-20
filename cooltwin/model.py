"""Lumped-parameter (2R2C-style) thermal model of a naturally ventilated hostel room.

The room is represented by two thermal nodes:

* ``Ti`` - indoor air and light contents (furniture, bedding)
* ``Tm`` - heavy building mass (roof slab and walls)

Energy balance (all temperatures in deg C, heat flows in W):

    C_i dTi/dt = (To - Ti)/R_ie + (Tm - Ti)/R_im + H_v(t) (To - Ti)
                 + g_win * I_sol * shade + Q_int
    C_m dTm/dt = (Ti - Tm)/R_im + (T_sa - Tm)/R_roof

where ``T_sa = To + alpha * I_sol / h_out`` is the sol-air temperature of the
roof surface. ``H_v`` is the ventilation conductance, which can differ between
day and night to represent opening windows after sunset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
import pandas as pd

AIR_DENSITY = 1.2  # kg/m3
AIR_CP = 1005.0  # J/(kg K)
H_OUT = 20.0  # W/(m2 K), combined outside surface coefficient


@dataclass(frozen=True)
class RoomParams:
    """Physical parameters of one room.

    Defaults describe a typical single room (about 3 m x 4 m x 3 m) on the top
    floor of an older RCC hostel block with an uninsulated roof slab.
    """

    R_ie: float = 0.033  # K/W, walls + windows, indoor air to outdoor air
    R_im: float = 0.0025  # K/W, indoor air to building mass
    R_roof: float = 0.035  # K/W, building mass to roof sol-air node
    C_i: float = 2.0e5  # J/K, air + light contents
    C_m: float = 1.0e7  # J/K, slab + walls
    roof_area: float = 12.0  # m2
    alpha_roof: float = 0.75  # solar absorptance of roof surface (-)
    g_win: float = 0.5  # m2, effective solar aperture of windows
    shade: float = 1.0  # fraction of window solar gain admitted (1 = unshaded)
    ach_day: float = 3.0  # air changes per hour during the day
    ach_night: float = 2.0  # air changes per hour at night (22:00-06:00)
    volume: float = 36.0  # m3
    q_internal: float = 150.0  # W, occupants, laptop, lights

    def with_updates(self, **changes: float) -> "RoomParams":
        return replace(self, **changes)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def ventilation_conductance(ach: float, volume: float) -> float:
    """Convert air changes per hour to a heat conductance in W/K."""
    return AIR_DENSITY * AIR_CP * volume * ach / 3600.0


def _is_night(hours: np.ndarray) -> np.ndarray:
    return (hours >= 22) | (hours < 6)


def simulate(
    weather: pd.DataFrame,
    params: RoomParams,
    t_init: float | None = None,
    substep_s: float = 120.0,
) -> pd.DataFrame:
    """Simulate indoor temperature for a weather time series.

    Parameters
    ----------
    weather:
        DataFrame indexed by timestamp with columns ``t_out`` (deg C) and
        ``solar`` (global horizontal irradiance, W/m2). The index must be
        regularly spaced.
    params:
        Room parameters.
    t_init:
        Starting temperature for both nodes. Defaults to the first outdoor
        temperature plus 2 K.
    substep_s:
        Internal integration step in seconds. The model is integrated with an
        explicit scheme, so the step must stay well below the smallest time
        constant (about R_im * C_i).

    Returns
    -------
    DataFrame with columns ``t_in`` (indoor air) and ``t_mass``.
    """
    if not {"t_out", "solar"}.issubset(weather.columns):
        raise ValueError("weather must contain 't_out' and 'solar' columns")
    if len(weather) < 2:
        raise ValueError("weather must contain at least two time steps")

    idx = weather.index
    step_s = float((idx[1] - idx[0]).total_seconds())

    t_out = weather["t_out"].to_numpy(dtype=float)
    solar = weather["solar"].to_numpy(dtype=float)
    night = _is_night(idx.hour.to_numpy())

    h_day = ventilation_conductance(params.ach_day, params.volume)
    h_night = ventilation_conductance(params.ach_night, params.volume)

    # Explicit integration is stable while dt stays well below the shortest
    # time constant of the air node, so refine the sub-step when needed.
    g_air = 1.0 / params.R_ie + 1.0 / params.R_im + max(h_day, h_night)
    tau_air = params.C_i / g_air
    dt_max = min(substep_s, 0.25 * tau_air)
    n_sub = max(1, int(np.ceil(step_s / dt_max)))
    dt = step_s / n_sub

    ti = t_out[0] + 2.0 if t_init is None else float(t_init)
    tm = ti
    t_in = np.empty_like(t_out)
    t_mass = np.empty_like(t_out)

    for k in range(len(t_out)):
        t_in[k], t_mass[k] = ti, tm
        h_v = h_night if night[k] else h_day
        t_sa = t_out[k] + params.alpha_roof * solar[k] / H_OUT
        q_sol_win = params.g_win * solar[k] * params.shade
        for _ in range(n_sub):
            q_i = (
                (t_out[k] - ti) / params.R_ie
                + (tm - ti) / params.R_im
                + h_v * (t_out[k] - ti)
                + q_sol_win
                + params.q_internal
            )
            q_m = (ti - tm) / params.R_im + (t_sa - tm) / params.R_roof
            ti += dt * q_i / params.C_i
            tm += dt * q_m / params.C_m

    return pd.DataFrame({"t_in": t_in, "t_mass": t_mass}, index=idx)
