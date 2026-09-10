from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rooftop_pv.physics import (
    LossFactors,
    PVConfig,
    _horizon_elevation,
    _time_step_hours,
    _validate_weather,
    simulate_pv,
)


def _config(**overrides: object) -> PVConfig:
    values: dict[str, object] = {
        "latitude": 47.3769,
        "longitude": 8.5417,
        "tilt_deg": 30.0,
        "azimuth_deg": 180.0,
        "available_area_m2": 20.0,
        "module_area_m2": 2.0,
        "module_efficiency": 0.20,
        "module_count": 10,
    }
    values.update(overrides)
    return PVConfig(**values)


def _weather(*, periods: int = 2, start: str = "2024-06-21 09:00", **overrides: object) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h", tz="Europe/Zurich")
    values: dict[str, object] = {
        "ghi": [800.0] * periods,
        "dni": [550.0] * periods,
        "dhi": [250.0] * periods,
        "temp_air": [20.0] * periods,
        "wind_speed": [2.0] * periods,
    }
    values.update(overrides)
    return pd.DataFrame(values, index=index)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("latitude", 91.0, "latitude"),
        ("latitude", np.nan, "latitude"),
        ("longitude", 181.0, "longitude"),
        ("longitude", np.inf, "longitude"),
        ("elevation_m", np.nan, "elevation_m"),
        ("tilt_deg", -1.0, "tilt_deg"),
        ("tilt_deg", 91.0, "tilt_deg"),
        ("azimuth_deg", np.nan, "azimuth_deg"),
        ("available_area_m2", -1.0, "available_area_m2"),
        ("available_area_m2", np.inf, "available_area_m2"),
        ("module_area_m2", 0.0, "module_area_m2"),
        ("module_area_m2", np.nan, "module_area_m2"),
        ("module_efficiency", 0.0, "module_efficiency"),
        ("module_efficiency", 1.1, "module_efficiency"),
        ("inverter_efficiency", 0.0, "inverter_efficiency"),
        ("inverter_reference_efficiency", np.inf, "inverter_reference_efficiency"),
        ("inverter_ac_power_w", -1.0, "inverter_ac_power_w"),
        ("inverter_ac_power_w", np.nan, "inverter_ac_power_w"),
        ("temperature_coefficient_per_deg_c", 0.0, "temperature_coefficient"),
        ("temperature_coefficient_per_deg_c", np.nan, "temperature_coefficient"),
        ("albedo", -0.01, "albedo"),
        ("albedo", 1.01, "albedo"),
        ("faiman_u0", 0.0, "faiman_u0"),
        ("faiman_u0", np.nan, "faiman_u0"),
        ("faiman_u1", -1.0, "faiman_u1"),
        ("timestep_hours", 0.0, "timestep_hours"),
        ("time_step_hours", np.nan, "timestep_hours"),
    ],
)
def test_pv_config_rejects_nonphysical_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**{field: value})


def test_pv_config_rejects_invalid_relationships_and_types() -> None:
    with pytest.raises(TypeError, match="losses"):
        _config(losses=object())
    with pytest.raises(ValueError, match="non-negative integer"):
        _config(module_count=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        _config(module_count=1.5)
    with pytest.raises(ValueError, match="exceeds"):
        _config(available_area_m2=0.0, module_count=1)
    with pytest.raises(ValueError, match="unsupported"):
        _config(transposition_model="unknown")
    with pytest.raises(ValueError, match="must agree"):
        _config(timestep_hours=1.0, time_step_hours=2.0)
    with pytest.raises(ValueError, match="monthly"):
        _config(monthly_losses={13: {"snow": 0.1}})


def test_pv_config_can_infer_zero_modules_and_expose_nameplate_limits() -> None:
    config = _config(available_area_m2=0.0, module_count=None)

    assert config.module_count == 0
    assert config.pdc0_w == 0.0
    assert config.inverter_limit_w == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("soiling", -0.01),
        ("snow", 1.01),
        ("wiring", np.nan),
        ("mismatch", np.inf),
        ("degradation", -0.01),
        ("availability", -0.01),
    ],
)
def test_loss_factors_reject_invalid_fractions(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field if field != "availability" else "availability"):
        LossFactors(**{field: value})


def test_loss_factor_monthly_validation_rejects_bad_keys_and_names() -> None:
    with pytest.raises(ValueError, match="calendar months"):
        LossFactors(monthly={0: {"snow": 0.1}})
    with pytest.raises(ValueError, match="unknown monthly"):
        LossFactors(monthly={1: {"curtailment": 0.1}})
    with pytest.raises(ValueError, match="snow"):
        LossFactors(monthly={1: {"snow": 2.0}})
    with pytest.raises(ValueError, match="either"):
        _config(losses=LossFactors(monthly={1: {"snow": 0.1}}), monthly_losses={1: {"snow": 0.2}})


@pytest.mark.parametrize(
    ("weather", "error"),
    [
        (object(), "pandas DataFrame"),
        (pd.DataFrame(), "DatetimeIndex"),
        (pd.DataFrame(index=pd.date_range("2024-01-01", periods=1, tz="UTC")), "required"),
        (_weather().iloc[0:0], "at least one row"),
        (_weather().set_axis([_weather().index[0], _weather().index[0]], axis="index"), "duplicate"),
    ],
)
def test_weather_shape_and_index_validation(weather: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _validate_weather(weather)  # type: ignore[arg-type]


@pytest.mark.parametrize("column", ["ghi", "dni", "dhi", "temp_air", "wind_speed"])
def test_weather_rejects_nonfinite_values(column: str) -> None:
    weather = _weather(**{column: [np.nan, 20.0]})

    with pytest.raises(ValueError, match=column):
        _validate_weather(weather)


@pytest.mark.parametrize("column", ["ghi", "dni", "dhi", "wind_speed"])
def test_weather_rejects_negative_irradiance_and_wind(column: str) -> None:
    weather = _weather(**{column: [-1.0, 20.0]})

    with pytest.raises(ValueError, match="non-negative"):
        _validate_weather(weather)


@pytest.mark.parametrize("value", [np.nan, -0.1, 1.1])
def test_weather_rejects_invalid_albedo(value: float) -> None:
    weather = _weather(albedo=[value, 0.2])

    with pytest.raises(ValueError, match="albedo"):
        _validate_weather(weather)


def test_weather_accepts_sortable_finite_values_and_preserves_albedo() -> None:
    weather = _weather(albedo=[0.3, 0.4]).iloc[::-1]

    result = _validate_weather(weather)

    assert result.index.is_monotonic_increasing
    assert result["albedo"].tolist() == pytest.approx([0.3, 0.4])


def test_time_step_validation_covers_singleton_and_nonincreasing_indices() -> None:
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    assert _time_step_hours(pd.DatetimeIndex([timestamp]), None).tolist() == [1.0]
    with pytest.raises(ValueError, match="increasing"):
        _time_step_hours(pd.DatetimeIndex([timestamp, timestamp]), None)


def test_time_step_rejects_nonpositive_forward_interval() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
        ]
    )

    with pytest.raises(ValueError, match="increasing"):
        _time_step_hours(index, None)


def test_time_step_rejects_large_interior_gap() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 12:00:00+00:00",
            "2024-01-01 13:00:00+00:00",
        ]
    )

    with pytest.raises(ValueError, match="large gap"):
        _time_step_hours(index, None)


@pytest.mark.parametrize(
    "profile",
    [
        {},
        {0.0: 1.0},
        {0.0: np.nan, 90.0: 1.0},
        {0.0: -1.0, 90.0: 1.0},
        {0.0: 91.0, 90.0: 1.0},
        {0.0: 1.0, 360.0: 2.0},
        {0.0: 1.0, 360.0: 1.0},
    ],
)
def test_horizon_mapping_validation(profile: dict[float, float]) -> None:
    with pytest.raises(ValueError, match="horizon"):
        _horizon_elevation(pd.Series([0.0]), profile)


def test_horizon_sequence_and_native_rows_validation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _horizon_elevation(pd.Series([0.0]), [1.0])
    with pytest.raises(ValueError, match="finite"):
        _horizon_elevation(pd.Series([0.0]), [1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        _horizon_elevation(pd.Series([0.0]), [{"A": 0.0, "H_hor": np.nan}, {"A": 90.0, "H_hor": 1.0}])
    with pytest.raises(ValueError, match=r"\[0, 90\]"):
        _horizon_elevation(pd.Series([0.0]), [{"A": 0.0, "H_hor": 91.0}, {"A": 90.0, "H_hor": 1.0}])
    with pytest.raises(ValueError, match="conflicting"):
        _horizon_elevation(
            pd.Series([0.0]),
            [{"A": -180.0, "H_hor": 1.0}, {"A": 180.0, "H_hor": 2.0}],
        )
    with pytest.raises(ValueError, match="unique"):
        _horizon_elevation(
            pd.Series([0.0]),
            [{"A": -180.0, "H_hor": 1.0}, {"A": 180.0, "H_hor": 1.0}],
        )
    values = _horizon_elevation(pd.Series([0.0, 180.0]), [1.0, 3.0])
    assert values.tolist() == pytest.approx([1.0, 3.0])


def test_simulation_rejects_wrong_config_and_zero_inverter_is_bounded() -> None:
    with pytest.raises(TypeError, match="PVConfig"):
        simulate_pv(_weather(), object())  # type: ignore[arg-type]

    result = simulate_pv(_weather(start="2024-06-21 00:00"), _config(inverter_ac_power_w=0.0))

    assert result.summary["energy_kwh"] == pytest.approx(0.0)
    assert np.all(result.hourly["ac_power_w"] == 0.0)
    assert result.summary_json == result.to_summary_json()
    assert math.isfinite(result.summary["energy_kwh"])
