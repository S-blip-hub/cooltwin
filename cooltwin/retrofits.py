"""Low-cost retrofit options and how to compare them on a calibrated room."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .model import RoomParams, simulate

OVERHEAT_THRESHOLD_C = 31.0
NIGHT_HOURS = (22, 7)  # sleep period, used for the night comfort metric


@dataclass(frozen=True)
class Retrofit:
    name: str
    description: str
    apply: Callable[[RoomParams], RoomParams]
    cost_inr: float  # per room; illustrative, replace with local quotes


def _cool_roof(p: RoomParams) -> RoomParams:
    # High-reflectance white coating; aged solar absorptance around 0.3.
    return p.with_updates(alpha_roof=min(p.alpha_roof, 0.3))


def _external_shading(p: RoomParams) -> RoomParams:
    # Fixed chajja or external blind cutting direct window gain by about 60 %.
    return p.with_updates(shade=p.shade * 0.4)


def _night_ventilation(p: RoomParams) -> RoomParams:
    # Window exhaust fan run after 22:00, raising night air changes to ~8 ACH.
    return p.with_updates(ach_night=max(p.ach_night, 8.0))


def _roof_insulation(p: RoomParams) -> RoomParams:
    # 50 mm XPS board over the slab adds roughly 1.5 m2K/W of resistance.
    return p.with_updates(R_roof=p.R_roof + 1.5 / p.roof_area)


DEFAULT_RETROFITS: List[Retrofit] = [
    Retrofit("cool_roof", "Reflective white roof coating", _cool_roof, 1800.0),
    Retrofit("external_shading", "External shading on the window", _external_shading, 3000.0),
    Retrofit("night_ventilation", "Exhaust fan for night flushing", _night_ventilation, 2500.0),
    Retrofit("roof_insulation", "50 mm XPS insulation over roof slab", _roof_insulation, 7200.0),
]


def comfort_metrics(t_in: pd.Series, threshold: float = OVERHEAT_THRESHOLD_C) -> Dict[str, float]:
    """Heat-exposure indicators for an indoor temperature series."""
    step_h = (t_in.index[1] - t_in.index[0]).total_seconds() / 3600.0
    hours = t_in.index.hour
    start, end = NIGHT_HOURS
    night = (hours >= start) | (hours < end)
    excess = np.clip(t_in.to_numpy() - threshold, 0.0, None)
    return {
        "peak_C": float(t_in.max()),
        "mean_night_C": float(t_in[night].mean()),
        "hours_above_threshold": float((excess > 0).sum() * step_h),
        "degree_hours_above_threshold": float(excess.sum() * step_h),
    }


def compare_retrofits(
    weather: pd.DataFrame,
    params: RoomParams,
    retrofits: List[Retrofit] | None = None,
    include_combined: bool = True,
    warmup_hours: float = 24.0,
) -> pd.DataFrame:
    """Simulate each retrofit on the calibrated room and rank them.

    Ranking is by reduction in overheating degree-hours per 1,000 rupees spent.
    """
    retrofits = list(retrofits or DEFAULT_RETROFITS)
    if include_combined and len(retrofits) > 1:
        cheapest_two = sorted(retrofits, key=lambda r: r.cost_inr)[:2]
        retrofits.append(
            Retrofit(
                "combined_low_cost",
                " + ".join(r.description for r in cheapest_two),
                lambda p, rs=cheapest_two: _chain(p, rs),
                sum(r.cost_inr for r in cheapest_two),
            )
        )

    keep = weather.index >= weather.index[0] + pd.Timedelta(hours=warmup_hours)
    base_t = simulate(weather, params)["t_in"][keep]
    base = comfort_metrics(base_t)

    rows = [{"option": "baseline", "description": "Existing room", "cost_inr": 0.0, **base}]
    for r in retrofits:
        t = simulate(weather, r.apply(params))["t_in"][keep]
        rows.append({"option": r.name, "description": r.description, "cost_inr": r.cost_inr, **comfort_metrics(t)})

    df = pd.DataFrame(rows).set_index("option")
    df["peak_reduction_K"] = base["peak_C"] - df["peak_C"]
    df["degree_hours_saved"] = base["degree_hours_above_threshold"] - df["degree_hours_above_threshold"]
    df["degree_hours_saved_per_1000_inr"] = np.where(
        df["cost_inr"] > 0, 1000.0 * df["degree_hours_saved"] / df["cost_inr"].where(df["cost_inr"] > 0), np.nan
    )
    ranked = df.drop(index="baseline").sort_values("degree_hours_saved_per_1000_inr", ascending=False)
    return pd.concat([df.loc[["baseline"]], ranked])


def _chain(p: RoomParams, rs: List[Retrofit]) -> RoomParams:
    for r in rs:
        p = r.apply(p)
    return p
