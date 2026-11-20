"""End-to-end demo on synthetic data.

1. Generate a hot spell and a "true" room whose parameters are hidden.
2. Simulate that room and add sensor noise to create a fake sensor log.
3. Calibrate the digital twin starting from generic default parameters.
4. Rank low-cost retrofits on the calibrated twin.
5. Save figures and tables to ./outputs.

Because the true parameters are known here, the demo also shows how close
calibration gets to them. With real sensors the true values are unknown,
which is exactly why calibration is needed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running from a fresh clone without installing

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cooltwin import RoomParams, calibrate, compare_retrofits, simulate, synthetic_hot_spell

OUT = ROOT / "outputs"
DATA = ROOT / "data"


def make_sensor_log(seed: int = 3):
    weather = synthetic_hot_spell(days=7, seed=11)
    true_room = RoomParams(
        R_ie=0.04, R_im=0.003, R_roof=0.05, C_i=3.0e5, C_m=1.5e7, alpha_roof=0.7, g_win=0.4,
        ach_day=4.0, ach_night=3.0,
    )
    t_in = simulate(weather, true_room)["t_in"]
    rng = np.random.default_rng(seed)
    noisy = t_in + rng.normal(0.0, 0.25, size=len(t_in))  # typical low-cost sensor noise
    log = weather.copy()
    log["t_in"] = noisy.round(2)
    return log, true_room


def main() -> None:
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    log, true_room = make_sensor_log()
    log.round(2).rename_axis("timestamp").to_csv(DATA / "sample_sensor_log.csv")

    result = calibrate(log)
    print("Calibration metrics:", {k: round(v, 3) for k, v in result.metrics.items()})
    print("\nParameter     true        fitted")
    for name, fitted in result.fitted.items():
        print(f"{name:<12}{getattr(true_room, name):>10.4g}  {fitted:>10.4g}")

    table = compare_retrofits(log[["t_out", "solar"]], result.params)
    table.round(3).to_csv(OUT / "retrofit_ranking.csv")
    print("\n", table[["cost_inr", "peak_C", "peak_reduction_K", "degree_hours_saved_per_1000_inr"]].round(2))

    # Individual parameters are not uniquely identifiable from temperature
    # alone, so check whether the twin still ranks retrofits the same way as
    # the true room would. That ranking is what a facility manager acts on.
    truth = compare_retrofits(log[["t_out", "solar"]], true_room)
    same = list(truth.index) == list(table.index)
    print("\nRanking on true room:      ", list(truth.drop(index="baseline").index))
    print("Ranking on calibrated twin:", list(table.drop(index="baseline").index))
    print("Same order:", same)

    # Figure 1: measured vs calibrated twin
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(log.index, log["t_out"], color="#9aa5b1", lw=1, label="Outdoor air")
    ax.plot(log.index, log["t_in"], color="#c0392b", lw=0.8, alpha=0.6, label="Indoor (sensor)")
    ax.plot(result.simulated.index, result.simulated, color="#1f4e79", lw=1.6, label="Indoor (calibrated twin)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Calibration on a 7-day hot spell  |  RMSE {result.metrics['rmse_K']:.2f} K")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)

    # Figure 2: retrofit ranking
    ranked = table.drop(index="baseline")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ranked.index[::-1], ranked["degree_hours_saved_per_1000_inr"][::-1], color="#1f4e79")
    ax.set_xlabel("Overheating degree-hours saved per ₹1,000")
    ax.set_title("Retrofit value for money (above 31 °C)")
    fig.tight_layout()
    fig.savefig(OUT / "retrofit_ranking.png", dpi=150)
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
