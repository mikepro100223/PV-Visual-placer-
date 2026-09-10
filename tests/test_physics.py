from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rooftop_pv.physics import LossFactors, PVConfig, _horizon_elevation, simulate_pv


def _weather(periods: int = 6) -> pd.DataFrame:
    index = pd.date_range("2024-06-21 09:00", periods=periods, freq="h", tz="Europe/Zurich")
    return pd.DataFrame(
        {
            "ghi": [800.0] * periods,
            "dni": [550.0] * periods,
            "dhi": [250.0] * periods,
            "temp_air": [20.0] * periods,
            "wind_speed": [2.0] * periods,
        },
        index=index,
    )


def _config(**overrides) -> PVConfig:
    values = dict(
        latitude=47.3769,
        longitude=8.5417,
        tilt_deg=30.0,
        azimuth_deg=180.0,
        available_area_m2=20.0,
        module_count=10,
        module_area_m2=2.0,
        module_efficiency=0.20,
        inverter_ac_power_w=1_000.0,
    )
    values.update(overrides)
    return PVConfig(**values)


def test_simulation_has_explicit_hourly_columns_and_integrates_energy():
    result = simulate_pv(_weather(), _config())

    assert len(result.hourly) == 6
    assert {"poa_global", "cell_temperature_c", "dc_power_w", "ac_power_w", "energy_kwh"}.issubset(
        result.hourly.columns
    )
    assert result.hourly["energy_kwh"].sum() == pytest.approx(
        result.summary["energy_kwh"]
    )
    assert result.summary["energy_kwh"] > 0
    assert result.summary["peak_ac_power_w"] <= 1_000.0 + 1e-9


def test_night_is_zero_even_if_weather_contains_nonzero_irradiance():
    weather = _weather(2)
    weather.index = pd.date_range("2024-06-21 00:00", periods=2, freq="h", tz="Europe/Zurich")

    result = simulate_pv(weather, _config())

    assert np.allclose(result.hourly[["poa_global", "dc_power_w", "ac_power_w"]], 0.0)


def test_inverter_clipping_is_bounded():
    result = simulate_pv(_weather(), _config(inverter_ac_power_w=50.0))

    assert result.hourly["ac_power_w"].max() <= 50.0 + 1e-9
    assert result.hourly["ac_power_w"].max() > 0.0
    assert result.summary["inverter_clipping_kwh"] >= 0.0
    assert result.summary["inverter_clipping_kwh"] > 0.0

    high_cap = simulate_pv(_weather(), _config(inverter_ac_power_w=100_000.0))
    assert high_cap.summary["inverter_clipping_kwh"] == pytest.approx(0.0)


def test_loss_breakdown_applies_each_factor_and_is_exposed():
    losses = LossFactors(soiling=0.1, snow=0.1, wiring=0.1, mismatch=0.1, degradation=0.1, availability=0.8)
    result = simulate_pv(_weather(), _config(losses=losses, inverter_ac_power_w=100_000.0))

    expected_factor = 0.9**5 * 0.8
    assert result.summary["loss_multiplier"] == pytest.approx(expected_factor)
    assert result.summary["losses"]["availability"] == pytest.approx(0.8)
    assert result.hourly["loss_multiplier"].iloc[0] == pytest.approx(expected_factor)


def test_monthly_losses_can_be_given_on_pv_config():
    result = simulate_pv(
        _weather(),
        _config(monthly_losses={6: {"soiling": 0.5, "availability": 1.0}}),
    )
    assert result.summary["losses"]["soiling"] == pytest.approx(0.5)
    assert result.summary["losses"]["availability"] == pytest.approx(1.0)


def test_horizon_profile_shades_direct_beam_only():
    unshaded = simulate_pv(_weather(), _config())
    shaded = simulate_pv(_weather(), _config(horizon_profile={az: 90.0 for az in range(0, 360, 10)}))

    assert (shaded.hourly["poa_direct_wm2"] == 0.0).all()
    assert shaded.summary["energy_kwh"] < unshaded.summary["energy_kwh"]


def test_native_pvgis_horizon_rows_and_closed_endpoints_are_supported():
    profile = [{"A": -180.0, "H_hor": 6.1}, {"A": 0.0, "H_hor": 4.0}, {"A": 180.0, "H_hor": 6.1}]
    result = simulate_pv(_weather(), _config(horizon_profile=profile))

    assert result.summary["energy_kwh"] >= 0.0


def test_native_pvgis_horizon_azimuths_are_converted_to_pvlib_convention():
    profile = [
        {"A": -180.0, "H_hor": 1.0},  # north
        {"A": -90.0, "H_hor": 2.0},   # east
        {"A": 0.0, "H_hor": 3.0},     # south
        {"A": 90.0, "H_hor": 4.0},    # west
        {"A": 180.0, "H_hor": 1.0},   # closed north endpoint
    ]
    horizon = _horizon_elevation(pd.Series([0.0, 90.0, 180.0, 270.0]), profile)

    assert np.allclose(horizon, [1.0, 2.0, 3.0, 4.0])


def test_weather_must_have_timezone_aware_index_and_required_columns():
    weather = _weather().copy()
    weather.index = weather.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        simulate_pv(weather, _config())

    with pytest.raises(ValueError, match="dhi"):
        simulate_pv(_weather().drop(columns="dhi"), _config())


def test_large_timestamp_gap_is_rejected_without_explicit_timestep():
    weather = _weather(2)
    weather.index = pd.DatetimeIndex(
        ["2024-06-21 07:00:00+00:00", "2024-06-28 07:00:00+00:00"]
    )
    with pytest.raises(ValueError, match="large gap"):
        simulate_pv(weather, _config())

    explicit = simulate_pv(weather, _config(time_step_hours=1.0))
    assert explicit.summary["yearly_energy_kwh"]


def test_irregular_samples_use_forward_interval_and_median_for_final_sample():
    weather = _weather(3)
    weather.index = pd.DatetimeIndex(
        [
            "2024-06-21 09:00:00+00:00",
            "2024-06-21 10:00:00+00:00",
            "2024-06-21 12:00:00+00:00",
        ]
    )

    result = simulate_pv(weather, _config())

    assert result.hourly["interval_hours"].tolist() == pytest.approx([1.0, 2.0, 1.5])
