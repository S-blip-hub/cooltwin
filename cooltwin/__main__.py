"""Command-line interface.

    python -m cooltwin run data/sample_sensor_log.csv --out outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibrate import calibrate
from .retrofits import compare_retrofits
from .weather import load_sensor_csv


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="cooltwin", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="calibrate on a sensor log and rank retrofits")
    run.add_argument("csv", help="CSV with timestamp, t_out, solar, t_in")
    run.add_argument("--out", default="outputs", help="output folder")
    args = parser.parse_args(argv)

    data = load_sensor_csv(args.csv)
    result = calibrate(data)
    table = compare_retrofits(data[["t_out", "solar"]], result.params)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(
        json.dumps({"fitted": result.fitted, "metrics": result.metrics}, indent=2)
    )
    table.to_csv(out / "retrofit_ranking.csv")

    print("Calibration fit:")
    for k, v in result.metrics.items():
        print(f"  {k:>14}: {v:7.3f}")
    print("\nRetrofit ranking (degree-hours above 31 C saved per Rs 1,000):")
    cols = ["cost_inr", "peak_C", "peak_reduction_K", "degree_hours_saved", "degree_hours_saved_per_1000_inr"]
    print(table[cols].round(2).to_string())


if __name__ == "__main__":
    main()
