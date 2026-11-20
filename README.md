# CoolTwin

**A sensor-calibrated digital twin for low-cost heat-resilience retrofits of hostel rooms.**

Hostel rooms in older RCC blocks across eastern India become dangerously hot in the pre-monsoon months. Most retrofit decisions are made by guesswork. CoolTwin turns a few days of cheap temperature-sensor data into a calibrated thermal model of a room. It then tests retrofits virtually and ranks them by how much overheating each one removes per rupee spent.

## How it works

1. **Measure.** Log indoor temperature in a room, together with outdoor temperature and solar irradiance, at 10-minute intervals.
2. **Model.** Each room is a lumped-parameter thermal network with two nodes: indoor air and the heavy mass of the roof slab and walls. Heat enters through the walls, the roof (via sol-air temperature), the windows, ventilation and occupants.
3. **Calibrate.** Uncertain parameters are fitted to the measured data with bounded nonlinear least squares in log space. These are resistances, capacitances, roof solar absorptance and window solar aperture. Fit quality is reported as RMSE, MAE, CV(RMSE) and NMBE.
4. **Compare retrofits.** Each option modifies the calibrated room and is simulated over the same weather. Options are ranked by overheating degree-hours above 31 °C saved per ₹1,000.

### Model

```
C_i dTi/dt = (To - Ti)/R_ie + (Tm - Ti)/R_im + H_v(t)(To - Ti) + g_win·I·shade + Q_int
C_m dTm/dt = (Ti - Tm)/R_im + (T_sa - Tm)/R_roof
T_sa       = To + α_roof·I / h_out
```

`Ti` is indoor air, `Tm` is building mass, `To` is outdoor air, `I` is solar irradiance and `H_v` is ventilation conductance (different by day and night).

### Retrofit options included

| Option | What it changes | Indicative cost per room |
|---|---|---|
| Cool roof | Roof solar absorptance → 0.3 | ₹1,800 |
| External shading | Window solar gain × 0.4 | ₹3,000 |
| Night ventilation | Night air changes → 8 ACH | ₹2,500 |
| Roof insulation | +1.5 m²K/W on the roof slab | ₹7,200 |
| Combined low-cost | The two cheapest options together | sum |

Costs are placeholders for illustration. Replace them with local quotes before using the ranking for decisions.

## Quick start

```bash
pip install -r requirements.txt

# End-to-end demo on synthetic data
# (writes data/sample_sensor_log.csv and figures in ./outputs)
python examples/run_demo.py

# Calibrate on your own sensor log and rank retrofits
python -m cooltwin run data/sample_sensor_log.csv --out outputs

# Tests
python -m pytest
```

The sensor log is a CSV with columns `timestamp, t_out, solar, t_in`. Short gaps of up to 30 minutes are interpolated.

## Demo results

The demo data is **synthetic**. A hidden "true" room is simulated over a generated 7-day hot spell (outdoor 28–40 °C) and 0.25 K of sensor noise is added. Because the true room is known, the demo can check the method:

- The calibrated twin reproduces the indoor temperature with an RMSE of about 0.25 K, which is the level of the added sensor noise.
- Individual parameters are not uniquely identifiable from temperature data alone; different combinations fit almost equally well. The ranking of retrofits, which is what a facility manager acts on, matches the ranking on the true room.

Retrofit ranking on the calibrated twin (demo run, last six days of the hot spell):

| Option | Cost (₹) | Peak indoor (°C) | Peak reduction (K) | Degree-hours saved per ₹1,000 |
|---|---:|---:|---:|---:|
| Baseline | 0 | 39.5 | – | – |
| Cool roof | 1,800 | 37.9 | 1.6 | 98.8 |
| Cool roof + night ventilation | 4,300 | 37.1 | 2.3 | 65.4 |
| Night ventilation | 2,500 | 38.6 | 0.9 | 49.6 |
| External shading | 3,000 | 38.6 | 0.8 | 27.1 |
| Roof insulation | 7,200 | 38.0 | 1.4 | 24.3 |

## Status and next steps

- [x] Thermal model, calibration engine, retrofit comparison, CLI and tests
- [ ] Deploy low-cost temperature and humidity sensors in a hostel block
- [ ] Calibrate on measured data and validate on a held-out week
- [ ] Pilot the top-ranked retrofits in sample rooms and measure the actual change
- [ ] Add humidity and a heat-stress index alongside dry-bulb temperature

## Project structure

```
cooltwin/
  model.py       two-node thermal model and simulator
  calibrate.py   parameter fitting and goodness-of-fit metrics
  retrofits.py   retrofit options, comfort metrics, ranking
  weather.py     sensor CSV loader and synthetic hot-spell generator
  __main__.py    command-line interface
examples/run_demo.py
tests/
```

## Author

Shrey Singh, B.Arch, Department of Architecture and Regional Planning, IIT Kharagpur.

## License

MIT
