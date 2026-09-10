"""Bounded, transparent rooftop PV simulation built on pvlib.

The public entry point is :func:`simulate_pv`.  It deliberately accepts
measured or supplied weather rather than silently fetching a weather source;
the output summary records the assumptions used by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pvlib
from pvlib import irradiance, inverter, location, pvsystem, temperature


_REQUIRED_WEATHER = frozenset({"ghi", "dni", "dhi", "temp_air", "wind_speed"})
_LOSS_NAMES = ("soiling", "snow", "wiring", "mismatch", "degradation", "availability")
_TRANSPOSITION_MODELS = frozenset(
    {"isotropic", "klucher", "haydavies", "reindl", "king", "perez", "perez-driesse"}
)


def _validate_fraction(value: float, name: str, *, availability: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        if availability:
            raise ValueError(f"{name} must be a finite factor in [0, 1]")
        raise ValueError(f"{name} must be a finite loss fraction in [0, 1]")
    return result


@dataclass(frozen=True)
class LossFactors:
    """System loss assumptions.

    Soiling, snow, wiring, mismatch, and degradation are loss fractions.  In
    contrast, ``availability`` is an operating factor: 0.99 means 99% of the
    otherwise available production remains.  ``monthly`` optionally overrides
    any of these values for a calendar month, with the same semantics.
    """

    soiling: float = 0.02
    snow: float = 0.00
    wiring: float = 0.02
    mismatch: float = 0.02
    degradation: float = 0.00
    availability: float = 0.99
    monthly: Mapping[int, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        for name in _LOSS_NAMES:
            _validate_fraction(getattr(self, name), name, availability=name == "availability")
        if self.monthly is not None:
            for month, overrides in self.monthly.items():
                if int(month) != month or not 1 <= int(month) <= 12:
                    raise ValueError("monthly loss keys must be calendar months 1..12")
                unknown = set(overrides) - set(_LOSS_NAMES)
                if unknown:
                    raise ValueError(f"unknown monthly loss names: {sorted(unknown)}")
                for name, value in overrides.items():
                    _validate_fraction(value, name, availability=name == "availability")

    def for_month(self, month: int) -> dict[str, float]:
        values = {name: float(getattr(self, name)) for name in _LOSS_NAMES}
        if self.monthly and month in self.monthly:
            values.update({name: float(value) for name, value in self.monthly[month].items()})
        return values

    @staticmethod
    def multiplier(values: Mapping[str, float]) -> float:
        return float(
            (1.0 - values["soiling"])
            * (1.0 - values["snow"])
            * (1.0 - values["wiring"])
            * (1.0 - values["mismatch"])
            * (1.0 - values["degradation"])
            * values["availability"]
        )


@dataclass(frozen=True)
class PVConfig:
    """Fixed-roof PV system and model assumptions.

    ``available_area_m2`` is the usable roof-plane area after geometry
    exclusions.  If ``module_count`` is omitted it is conservatively inferred
    as ``floor(available_area_m2 / module_area_m2)``.  Module STC power is
    derived from area and efficiency at 1000 W/m², so this is a planning model
    rather than a substitute for a module datasheet.
    """

    latitude: float
    longitude: float
    tilt_deg: float
    azimuth_deg: float
    available_area_m2: float
    module_area_m2: float = 1.9
    module_efficiency: float = 0.20
    module_count: int | None = None
    inverter_ac_power_w: float | None = None
    inverter_efficiency: float = 0.96
    temperature_coefficient_per_deg_c: float = -0.004
    albedo: float = 0.20
    transposition_model: str = "haydavies"
    faiman_u0: float = 25.0
    faiman_u1: float = 6.84
    temp_ref_c: float = 25.0
    timestep_hours: float | None = None
    time_step_hours: float | None = None
    horizon_profile: Mapping[float, float] | Sequence[float] | None = None
    losses: LossFactors = field(default_factory=LossFactors)
    weather_source: str = "supplied_weather"
    elevation_m: float = 0.0
    inverter_reference_efficiency: float = 0.9637
    monthly_losses: Mapping[int, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.losses, LossFactors):
            raise TypeError("losses must be a LossFactors instance")
        if self.monthly_losses is not None:
            if self.losses.monthly is not None:
                raise ValueError("set monthly losses either on LossFactors or PVConfig, not both")
            object.__setattr__(
                self,
                "losses",
                LossFactors(
                    soiling=self.losses.soiling,
                    snow=self.losses.snow,
                    wiring=self.losses.wiring,
                    mismatch=self.losses.mismatch,
                    degradation=self.losses.degradation,
                    availability=self.losses.availability,
                    monthly=self.monthly_losses,
                ),
            )
        latitude = float(self.latitude)
        longitude = float(self.longitude)
        tilt = float(self.tilt_deg)
        azimuth = float(self.azimuth_deg)
        area = float(self.available_area_m2)
        module_area = float(self.module_area_m2)
        efficiency = float(self.module_efficiency)
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("latitude must be finite and in [-90, 90]")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("longitude must be finite and in [-180, 180]")
        if not math.isfinite(float(self.elevation_m)):
            raise ValueError("elevation_m must be finite")
        if not math.isfinite(tilt) or not 0 <= tilt <= 90:
            raise ValueError("tilt_deg must be finite and in [0, 90]")
        if not math.isfinite(azimuth):
            raise ValueError("azimuth_deg must be finite")
        if not math.isfinite(area) or area < 0:
            raise ValueError("available_area_m2 must be finite and non-negative")
        if not math.isfinite(module_area) or module_area <= 0:
            raise ValueError("module_area_m2 must be finite and positive")
        if not math.isfinite(efficiency) or not 0 < efficiency <= 1:
            raise ValueError("module_efficiency must be finite and in (0, 1]")
        if self.module_count is not None:
            if int(self.module_count) != self.module_count or int(self.module_count) < 0:
                raise ValueError("module_count must be a non-negative integer")
            count = int(self.module_count)
        else:
            count = math.floor(area / module_area)
            object.__setattr__(self, "module_count", count)
        if count * module_area > area + 1e-8:
            raise ValueError("module_count × module_area_m2 exceeds available_area_m2")
        inv_eff = float(self.inverter_efficiency)
        if not math.isfinite(inv_eff) or not 0 < inv_eff <= 1:
            raise ValueError("inverter_efficiency must be finite and in (0, 1]")
        inv_ref_eff = float(self.inverter_reference_efficiency)
        if not math.isfinite(inv_ref_eff) or not 0 < inv_ref_eff <= 1:
            raise ValueError("inverter_reference_efficiency must be finite and in (0, 1]")
        if self.inverter_ac_power_w is not None:
            inverter_ac = float(self.inverter_ac_power_w)
            if not math.isfinite(inverter_ac) or inverter_ac < 0:
                raise ValueError("inverter_ac_power_w must be finite and non-negative")
        gamma = float(self.temperature_coefficient_per_deg_c)
        if not math.isfinite(gamma) or gamma >= 0:
            raise ValueError("temperature_coefficient_per_deg_c must be finite and negative")
        albedo = float(self.albedo)
        if not math.isfinite(albedo) or not 0 <= albedo <= 1:
            raise ValueError("albedo must be finite and in [0, 1]")
        if self.transposition_model not in _TRANSPOSITION_MODELS:
            raise ValueError(f"unsupported transposition_model: {self.transposition_model}")
        if not math.isfinite(float(self.faiman_u0)) or float(self.faiman_u0) <= 0:
            raise ValueError("faiman_u0 must be finite and positive")
        if not math.isfinite(float(self.faiman_u1)) or float(self.faiman_u1) < 0:
            raise ValueError("faiman_u1 must be finite and non-negative")
        if self.timestep_hours is not None and self.time_step_hours is not None:
            if not math.isclose(float(self.timestep_hours), float(self.time_step_hours)):
                raise ValueError("timestep_hours and time_step_hours must agree when both are set")
        configured_timestep = self.timestep_hours
        if configured_timestep is None:
            configured_timestep = self.time_step_hours
        if configured_timestep is not None:
            dt = float(configured_timestep)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("timestep_hours must be finite and positive")
            object.__setattr__(self, "timestep_hours", dt)
            object.__setattr__(self, "time_step_hours", dt)

    @property
    def pdc0_w(self) -> float:
        """Array nameplate power at 1000 W/m² and 25 °C."""

        return float(self.module_count or 0) * self.module_area_m2 * self.module_efficiency * 1000.0

    @property
    def inverter_limit_w(self) -> float:
        """Inverter DC input limit corresponding to the configured AC cap."""

        ac = self.inverter_ac_power_w
        if ac is None:
            ac = self.pdc0_w * self.inverter_efficiency
        return float(ac) / self.inverter_efficiency if self.inverter_efficiency else 0.0


@dataclass(frozen=True)
class PVResult:
    """Simulation output: hourly values and a JSON-serialisable summary."""

    hourly: pd.DataFrame
    summary: dict[str, Any]

    @property
    def summary_json(self) -> str:
        return json.dumps(self.summary, indent=2, sort_keys=True)

    def to_summary_json(self) -> str:
        return self.summary_json


def _validate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a pandas DataFrame")
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise ValueError("weather index must be a pandas DatetimeIndex")
    if weather.index.tz is None:
        raise ValueError("weather index must be timezone-aware")
    if len(weather) == 0:
        raise ValueError("weather must contain at least one row")
    if weather.index.has_duplicates:
        raise ValueError("weather index must not contain duplicate timestamps")
    missing = _REQUIRED_WEATHER - set(weather.columns)
    if missing:
        raise ValueError(f"weather is missing required columns: {', '.join(sorted(missing))}")
    result = weather.sort_index().copy()
    for column in _REQUIRED_WEATHER:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"weather column {column!r} must contain finite numeric values")
        if column in {"ghi", "dni", "dhi", "wind_speed"} and (values < 0).any():
            raise ValueError(f"weather column {column!r} must be non-negative")
        result[column] = values
    if "albedo" in result:
        albedo = pd.to_numeric(result["albedo"], errors="coerce")
        if albedo.isna().any() or (albedo < 0).any() or (albedo > 1).any():
            raise ValueError("weather albedo must be finite and in [0, 1]")
        result["albedo"] = albedo
    return result


def _time_step_hours(index: pd.DatetimeIndex, configured: float | None) -> np.ndarray:
    if configured is not None:
        return np.full(len(index), float(configured), dtype=float)
    if len(index) == 1:
        return np.ones(1, dtype=float)
    # Weather samples conventionally describe the interval beginning at their
    # timestamp.  Use the forward interval to the next observation; the final
    # sample receives the median cadence because no next timestamp exists.
    differences = (index[1:] - index[:-1]).total_seconds() / 3600.0
    positive = differences[differences > 0]
    if len(positive) == 0:
        raise ValueError("weather timestamps must be increasing")
    first = float(np.median(positive))
    # A long unobserved period must not be interpreted as continuous generation
    # at the next observed power.  Explicit ``timestep_hours`` is the opt-in
    # for callers that have already interpolated or otherwise handled gaps.
    if first > 6.0:
        raise ValueError(
            "weather timestamps contain a large gap; supply timestep_hours only after handling it"
        )
    if (positive > max(3.0 * first, first + 6.0)).any():
        raise ValueError(
            "weather timestamps contain a large gap; supply timestep_hours only after handling it"
        )
    intervals = np.empty(len(index), dtype=float)
    intervals[:-1] = differences
    intervals[-1] = first
    if not np.isfinite(intervals).all() or (intervals <= 0).any():
        raise ValueError("weather timestamps must be increasing")
    return intervals


def _horizon_elevation(azimuth: pd.Series, profile: Mapping[float, float] | Sequence[float]) -> np.ndarray:
    if isinstance(profile, Mapping):
        if len(profile) < 2:
            raise ValueError("horizon_profile mapping needs at least two azimuth samples")
        points = sorted((float(key) % 360.0, float(value)) for key, value in profile.items())
        if any(not math.isfinite(key) or not math.isfinite(value) for key, value in points):
            raise ValueError("horizon_profile values must be finite")
        if any(value < 0 or value > 90 for _, value in points):
            raise ValueError("horizon elevations must be in [0, 90] degrees")
        # PVGIS returns both -180° and +180° for a closed horizon profile.
        # They are the same circular direction; accept that endpoint pair only
        # when it carries a consistent height.
        unique_points: list[tuple[float, float]] = []
        for angle, value in points:
            if unique_points and math.isclose(angle, unique_points[-1][0], abs_tol=1e-12):
                if not math.isclose(value, unique_points[-1][1], rel_tol=0, abs_tol=1e-9):
                    raise ValueError("horizon_profile has conflicting duplicate azimuths")
                continue
            unique_points.append((angle, value))
        angles = np.array([point[0] for point in unique_points], dtype=float)
        values = np.array([point[1] for point in unique_points], dtype=float)
        if len(angles) < 2:
            raise ValueError("horizon_profile mapping needs at least two unique azimuths")
        angles_ext = np.r_[angles[-1] - 360.0, angles, angles[0] + 360.0]
        values_ext = np.r_[values[-1], values, values[0]]
    else:
        profile_values = list(profile)
        if profile_values and all(isinstance(value, Mapping) for value in profile_values):
            # PVGIS horizon rows use a different azimuth convention from
            # pvlib: 0°=south, +90°=west, -90°=east, ±180°=north.
            # Convert native rows to the public standard 0°=north,
            # 90°=east, 180°=south, 270°=west before interpolation.
            points = sorted(
                ((float(value["A"]) + 180.0) % 360.0, float(value["H_hor"]))
                for value in profile_values
            )
            if any(not math.isfinite(angle) or not math.isfinite(value) for angle, value in points):
                raise ValueError("horizon_profile values must be finite")
            if any(value < 0 or value > 90 for _, value in points):
                raise ValueError("horizon elevations must be in [0, 90] degrees")
            unique_points = []
            for angle, value in points:
                if unique_points and math.isclose(angle, unique_points[-1][0], abs_tol=1e-12):
                    if not math.isclose(value, unique_points[-1][1], rel_tol=0, abs_tol=1e-9):
                        raise ValueError("horizon_profile has conflicting duplicate azimuths")
                    continue
                unique_points.append((angle, value))
            if len(unique_points) < 2:
                raise ValueError("horizon_profile needs at least two unique azimuths")
            angles = np.array([point[0] for point in unique_points], dtype=float)
            values = np.array([point[1] for point in unique_points], dtype=float)
            angles_ext = np.r_[angles[-1] - 360.0, angles, angles[0] + 360.0]
            values_ext = np.r_[values[-1], values, values[0]]
            az = np.asarray(azimuth, dtype=float) % 360.0
            return np.interp(az, angles_ext, values_ext)
        values = np.asarray(profile_values, dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("horizon_profile sequence needs at least two samples")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 90).any():
            raise ValueError("horizon elevations must be finite and in [0, 90] degrees")
        angles = np.arange(len(values), dtype=float) * 360.0 / len(values)
        angles_ext = np.r_[angles[-1] - 360.0, angles, angles[0] + 360.0]
        values_ext = np.r_[values[-1], values, values[0]]
    az = np.asarray(azimuth, dtype=float) % 360.0
    return np.interp(az, angles_ext, values_ext)


def _loss_series(index: pd.DatetimeIndex, losses: LossFactors) -> tuple[pd.Series, dict[str, float]]:
    rows = [losses.for_month(int(month)) for month in index.month]
    multiplier = pd.Series(
        [LossFactors.multiplier(row) for row in rows], index=index, dtype=float, name="loss_multiplier"
    )
    # Report the factors for the first simulated month as the base accounting;
    # monthly overrides are also retained in the hourly columns.
    return multiplier, rows[0]


def simulate_pv(weather: pd.DataFrame, config: PVConfig) -> PVResult:
    """Simulate hourly AC PV production from timezone-aware weather data.

    Required weather columns are ``ghi``, ``dni``, ``dhi``, ``temp_air``, and
    ``wind_speed``.  Irradiance is W/m², temperatures are °C, wind is m/s,
    powers are W, and ``energy_kwh`` uses each sample's actual timestamp
    interval (including daylight-saving transitions).
    """

    if not isinstance(config, PVConfig):
        raise TypeError("config must be a PVConfig")
    data = _validate_weather(weather)
    index = data.index
    loc = location.Location(
        latitude=config.latitude,
        longitude=config.longitude,
        tz=str(index.tz),
        altitude=config.elevation_m,
    )
    solar = loc.get_solarposition(index)
    albedo: float | pd.Series = data["albedo"] if "albedo" in data else config.albedo
    poa = irradiance.get_total_irradiance(
        surface_tilt=config.tilt_deg,
        surface_azimuth=config.azimuth_deg % 360.0,
        solar_zenith=solar["apparent_zenith"],
        solar_azimuth=solar["azimuth"],
        dni=data["dni"],
        ghi=data["ghi"],
        dhi=data["dhi"],
        dni_extra=irradiance.get_extra_radiation(index),
        albedo=albedo,
        model=config.transposition_model,
    )
    poa_direct = pd.Series(poa["poa_direct"], index=index, dtype=float).clip(lower=0)
    poa_diffuse = pd.Series(poa["poa_diffuse"], index=index, dtype=float).clip(lower=0)
    day = pd.Series(solar["apparent_elevation"] > 0, index=index)
    direct_shading = pd.Series(1.0, index=index, dtype=float)
    if config.horizon_profile is not None:
        horizon = _horizon_elevation(solar["azimuth"], config.horizon_profile)
        sun_elevation = np.asarray(solar["apparent_elevation"], dtype=float)
        direct_shading = pd.Series((sun_elevation > horizon).astype(float), index=index)
    poa_direct = poa_direct * direct_shading
    # Irradiance at night is explicitly zero even if supplied weather has
    # stale non-zero values; diffuse skylight is not horizon-shaded.
    poa_direct = poa_direct.where(day, 0.0)
    poa_diffuse = poa_diffuse.where(day, 0.0)
    poa_global = (poa_direct + poa_diffuse).clip(lower=0.0)

    cell_temperature = pd.Series(
        temperature.faiman(
            poa_global,
            data["temp_air"],
            data["wind_speed"],
            u0=config.faiman_u0,
            u1=config.faiman_u1,
        ),
        index=index,
        dtype=float,
    )
    raw_dc = pd.Series(
        pvsystem.pvwatts_dc(
            poa_global,
            cell_temperature,
            pdc0=config.pdc0_w,
            gamma_pdc=config.temperature_coefficient_per_deg_c,
            temp_ref=config.temp_ref_c,
        ),
        index=index,
        dtype=float,
    ).clip(lower=0.0)
    loss_multiplier, first_loss = _loss_series(index, config.losses)
    dc_power = (raw_dc * loss_multiplier).clip(lower=0.0)
    # PVWatts' efficiency curve is not intended to be extrapolated to very
    # large zeta values.  Bound the DC input at the inverter's DC limit before
    # evaluating it; otherwise an intentionally tiny AC inverter can produce
    # negative-then-zero power instead of a physically clipped output.
    inverter_dc_limit = config.inverter_limit_w
    inverter_dc_input = dc_power.clip(upper=inverter_dc_limit)
    ac_power = pd.Series(
        inverter.pvwatts(
            inverter_dc_input,
            pdc0=inverter_dc_limit,
            eta_inv_nom=config.inverter_efficiency,
            eta_inv_ref=config.inverter_reference_efficiency,
        ),
        index=index,
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    ac_cap = float(config.inverter_ac_power_w) if config.inverter_ac_power_w is not None else config.pdc0_w * config.inverter_efficiency
    # Re-evaluate the same curve without its hard AC cap so conversion loss is
    # not mislabeled as clipping.  Clamp zeta at one because PVWatts is not
    # defined as an efficiency extrapolation far above the inverter's DC
    # rating; the excess DC then appears as a positive clipping residual.
    if inverter_dc_limit > 0:
        zeta = np.minimum(dc_power.to_numpy(dtype=float) / inverter_dc_limit, 1.0)
        eta = config.inverter_efficiency / config.inverter_reference_efficiency * (
            -0.0162 * zeta
            - np.divide(0.0059, zeta, out=np.zeros_like(zeta), where=zeta != 0)
            + 0.9858
        )
        unclipped_ac = pd.Series(
            np.maximum(0.0, eta * dc_power.to_numpy(dtype=float)),
            index=index,
        )
    else:
        unclipped_ac = pd.Series(0.0, index=index)
    # ``pvwatts`` clips at ac_cap by design; this residual is true AC clipping
    # after the nonlinear part-load conversion curve has been accounted for.
    clipping = (unclipped_ac - ac_power).clip(lower=0.0)
    dt_hours = _time_step_hours(index, config.timestep_hours)
    energy_kwh = ac_power * dt_hours / 1000.0
    monthly = energy_kwh.groupby(energy_kwh.index.strftime("%Y-%m")).sum()
    yearly = energy_kwh.groupby(energy_kwh.index.strftime("%Y")).sum()
    summary: dict[str, Any] = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "tilt_deg": config.tilt_deg,
        "azimuth_deg": config.azimuth_deg % 360.0,
        "available_area_m2": config.available_area_m2,
        "module_count": config.module_count,
        "module_area_m2": config.module_area_m2,
        "module_efficiency": config.module_efficiency,
        "pdc0_w": config.pdc0_w,
        "inverter_ac_power_w": ac_cap,
        "elevation_m": config.elevation_m,
        "weather_source": config.weather_source,
        "energy_kwh": float(energy_kwh.sum()),
        "peak_ac_power_w": float(ac_power.max()),
        "inverter_clipping_kwh": float((clipping * dt_hours / 1000.0).sum()),
        "loss_multiplier": float(loss_multiplier.iloc[0]),
        "losses": first_loss,
        "monthly_energy_kwh": {key: float(value) for key, value in monthly.items()},
        "yearly_energy_kwh": {key: float(value) for key, value in yearly.items()},
        "assumptions": {
            "irradiance_transposition": config.transposition_model,
            "temperature_model": "pvlib.temperature.faiman",
            "horizon_profile": config.horizon_profile is not None,
            "horizon_shades": "direct beam only; diffuse sky and ground reflection are not horizon-shaded",
            "aoi_and_spectral_losses": "not modeled without module-specific parameters",
            "coordinates": "latitude/longitude in WGS84; roof geometry area is supplied separately",
        },
        "pvlib_version": getattr(pvlib, "__version__", "unknown"),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    hourly = pd.DataFrame(
        {
            "solar_elevation_deg": np.asarray(solar["apparent_elevation"], dtype=float),
            "solar_azimuth_deg": np.asarray(solar["azimuth"], dtype=float),
            "direct_shading_factor": direct_shading,
            "poa_direct_wm2": poa_direct,
            "poa_diffuse_wm2": poa_diffuse,
            "poa_global": poa_global,
            "cell_temperature_c": cell_temperature,
            "raw_dc_power_w": raw_dc,
            "loss_multiplier": loss_multiplier,
            "dc_power_w": dc_power,
            "ac_power_w": ac_power,
            "inverter_clipping_w": clipping,
            "interval_hours": dt_hours,
            "energy_kwh": energy_kwh,
        },
        index=index,
    )
    return PVResult(hourly=hourly, summary=summary)


__all__ = ["LossFactors", "PVConfig", "PVResult", "simulate_pv"]
