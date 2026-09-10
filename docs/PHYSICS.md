# Rooftop PV model

The implementation in `code/rooftop_pv/physics.py` is a bounded planning
model. It consumes a timezone-aware `pandas.DataFrame` with measured or
otherwise supplied hourly `ghi`, `dni`, `dhi` (W/m²), `temp_air` (°C), and
`wind_speed` (m/s). It does not silently download weather or infer roof slope
from RGB imagery.

## Model chain

`simulate_pv(weather, PVConfig(...))` performs the following steps for each
timestamp:

1. Compute apparent solar position from the configured WGS84 latitude and
   longitude.
2. Transpose horizontal irradiance to the fixed roof plane with
   `pvlib.irradiance.get_total_irradiance`. The configured albedo contributes
   ground-reflected irradiance. The default sky model is Hay-Davies.
3. Set direct, diffuse, and total plane-of-array irradiance to zero whenever
   the apparent solar elevation is at or below zero. An optional horizon
   profile (mapping azimuth degrees to horizon elevation degrees, a regularly
   sampled sequence over 360°, or native PVGIS rows with `A` and `H_hor`)
   attenuates the direct beam only. Native PVGIS azimuths are converted from
   its `0°=south, +90°=west, −90°=east` convention to the public
   `0°=north, 90°=east, 180°=south, 270°=west` convention. The closed PVGIS
   `-180°`/`180°` endpoint pair is treated as one circular direction.
   Diffuse sky and ground reflection are deliberately not skyline-shaded.
4. Estimate cell temperature using `pvlib.temperature.faiman` with explicit
   wind and irradiance inputs.
5. Compute raw DC power with the PVWatts equation:

   `Pdc = (POA / 1000) × Pdc0 × (1 + gamma × (Tcell − 25 °C))`

   where `Pdc0 = module_count × module_area × module_efficiency × 1000 W`.
6. Apply the loss multiplier and convert DC to AC using the PVWatts inverter
   model. The DC input is bounded at the inverter's DC limit before evaluating
   the efficiency curve, avoiding invalid extrapolation for deliberately
   undersized-inverter scenarios. The AC output is bounded by the configured
   inverter AC rating; clipping is reported separately from the inverter's
   nonlinear part-load conversion loss.
7. Integrate each power sample using the actual interval to the next timestamp
   (the final sample uses the median positive interval; an explicit
   `PVConfig.timestep_hours` can be used for already-resampled data). This
   preserves DST transitions instead of assuming that every calendar day has
   24 samples. Large gaps are rejected unless an explicit timestep is supplied
   after the caller has handled the missing observations.

The pvlib references for these calls are [total irradiance]
(https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html),
[Faiman temperature]
(https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html),
[PVWatts DC]
(https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html),
and [PVWatts inverter]
(https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html).

## Losses and uncertainty

`LossFactors` treats soiling, snow, wiring, mismatch, and degradation as
fractions removed from production. `availability` is different: it is a
remaining operating factor (`0.99` means 99% availability). A `monthly`
mapping can override any of these values by calendar month. Loss assumptions
are reported in `PVResult.summary`, including the combined multiplier, rather
than being hidden in an unexplained derate.

Monthly overrides can be supplied as `LossFactors(monthly=...)` or through the
convenience `PVConfig(monthly_losses=...)` field (choose one, not both).

This model does not include module-specific angle-of-incidence or spectral
losses, bifacial gain, row-to-row shading, electrical string mismatch, snow
accumulation dynamics, or measured inverter curves. Those require module,
mounting, layout, or monitoring data. The summary records the omitted AOI and
spectral losses explicitly.

## Geometry and area basis

`code/rooftop_pv/geometry.py` expects Shapely polygons in a planar metric CRS.
The caller should project geographic coordinates first (Swiss LV95 is a
reasonable choice for Swiss data). `calculate_usable_area` unions all
obstacles before subtraction, clips them to the roof through the difference,
and applies a perimeter setback. A half-pixel erosion can be requested with
`pixel_size_m` to avoid crediting uncertain raster-edge pixels.

`AreaResult.planimetric_area_m2` is the map footprint. The
`tilted_area_m2`/`usable_area_m2` value is `planimetric / cos(tilt)` and is the
area basis used for module-area planning. `excluded_area_m2` is the difference
from the original roof footprint, including requested setback and pixel
erosion. `place_panels` is intentionally
conservative: it returns only axis-aligned rectangles fully covered by the
usable geometry. It is not a construction layout and does not infer a slope
or orientation.

## Example

```python
from rooftop_pv.physics import LossFactors, PVConfig, simulate_pv

config = PVConfig(
    latitude=47.3769,
    longitude=8.5417,
    elevation_m=400,
    tilt_deg=30,
    azimuth_deg=180,             # north=0, east=90, south=180, west=270
    available_area_m2=42.0,
    module_area_m2=1.9,
    module_efficiency=0.21,
    # Omit module_count to use floor(available_area / module_area).
    inverter_ac_power_w=8_000,
    losses=LossFactors(soiling=0.02, wiring=0.02, availability=0.99),
)
result = simulate_pv(weather, config)
print(result.summary_json)
hourly = result.hourly
```

`PVResult.summary` contains both `monthly_energy_kwh` and
`yearly_energy_kwh` mappings. A reported value is the energy represented by
the supplied weather window; it is not an annualized claim unless the input
weather covers a full year.

The output is an estimate for scenario comparison. It is not a structural,
electrical, fire-safety, grid-connection, or permitting determination.
## Dach- und Modulebene

Das Dashboard verwendet derzeit dachparallele Montage: Dachneigung ist zugleich
Modulneigung. Für ein Flachdach darf die geometrische Dachneigung nicht erhöht
werden, nur um aufgeständerte Module zu simulieren; sonst würde die nutzbare
Dachfläche fälschlich durch `1 / cos(Neigung)` vergrössert. Aufständerung,
Reihenabstände und gegenseitige Verschattung erfordern eine separate Layoutplanung.
Das Rechenmodul `PVConfig` kann eine bereits unabhängig bestimmte Modulfläche
mit frei vorgegebener Modulneigung simulieren, ersetzt aber diese Planung nicht.
