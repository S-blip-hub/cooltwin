"""CoolTwin: a sensor-calibrated digital twin for heat-resilience retrofits of hostel rooms."""

from .calibrate import CalibrationResult, calibrate, fit_metrics
from .model import RoomParams, simulate
from .retrofits import DEFAULT_RETROFITS, Retrofit, comfort_metrics, compare_retrofits
from .weather import load_sensor_csv, synthetic_hot_spell

__version__ = "0.1.0"

__all__ = [
    "RoomParams",
    "simulate",
    "calibrate",
    "fit_metrics",
    "CalibrationResult",
    "Retrofit",
    "DEFAULT_RETROFITS",
    "comfort_metrics",
    "compare_retrofits",
    "load_sensor_csv",
    "synthetic_hot_spell",
]
