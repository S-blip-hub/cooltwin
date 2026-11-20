"""Weather inputs: loading measured data and generating synthetic hot-season days."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_hot_spell(
    start: str = "2026-05-10",
    days: int = 7,
    freq: str = "10min",
    t_min: float = 28.0,
    t_max: float = 39.0,
    peak_solar: float = 900.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate an idealised pre-monsoon hot spell for testing and demos.

    Outdoor temperature follows a daily cycle with the minimum near 03:00 and
    the maximum near 15:00. Solar irradiance is a half-sine between 06:00 and
    18:00, scaled each day by a random cloud factor.

    This is NOT measured data. It exists so the pipeline can be exercised end
    to end before sensors are deployed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=int(days * 24 * 60 / _minutes(freq)), freq=freq)
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0

    mean = 0.5 * (t_min + t_max)
    amp = 0.5 * (t_max - t_min)
    t_out = mean - amp * np.cos(2 * np.pi * (hours - 3.0) / 24.0)
    # Day-to-day variation, interpolated so there are no jumps at midnight.
    day_offset = rng.normal(0.0, 0.8, size=days + 1)
    day_pos = np.arange(len(idx)) / (len(idx) / days)
    t_out = t_out + np.interp(day_pos, np.arange(days + 1), day_offset)

    daylight = (hours >= 6) & (hours <= 18)
    solar = np.where(daylight, np.sin(np.pi * (hours - 6) / 12.0), 0.0)
    cloud = rng.uniform(0.75, 1.0, size=days)
    solar = peak_solar * solar * np.repeat(cloud, len(idx) // days)

    return pd.DataFrame({"t_out": t_out, "solar": np.clip(solar, 0, None)}, index=idx)


def load_sensor_csv(path: str) -> pd.DataFrame:
    """Load a sensor log with columns ``timestamp, t_out, solar, t_in``.

    Rows are sorted, duplicate timestamps dropped and short gaps (up to 30
    minutes) interpolated. The returned frame is indexed by timestamp.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "t_out", "solar", "t_in"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    freq = pd.infer_freq(df.index[:20]) or "10min"
    df = df.asfreq(freq)
    limit = max(1, int(30 / _minutes(freq)))
    return df.interpolate(limit=limit).dropna()


def _minutes(freq: str) -> float:
    return pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds() / 60.0
