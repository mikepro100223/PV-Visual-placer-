# Live Swiss rooftop-PV sources

`code/rooftop_pv/sources.py` is the narrow source boundary used by the app.
It does not invent a roof, image, or weather value when an upstream source is
empty or malformed. Every live request has a 5-second connect timeout, a
30-second read timeout, and at most two retries. Orthophoto requests are
bounded to 2,048 x 2,048 pixels, four million pixels total, and a maximum
5,000 m ground span.

## Public API

```python
from rooftop_pv.sources import (
    geocode, fetch_roofs, fetch_orthophoto, fetch_weather, fetch_horizon,
)

address = geocode("Hauptstrasse 1 5200 Brugg")
roofs = fetch_roofs(address.latitude, address.longitude)
roof = roofs[0]
image = fetch_orthophoto(roof_bbox2056, 1024, 1024)
weather, weather_meta = fetch_weather(address.latitude, address.longitude)
```

The result records are intentionally small and serialisable:

- `GeocodeResult` contains the first GeoAdmin result in WGS84 (`latitude`,
  `longitude`) and derived LV95 (`easting`, `northing`) coordinates.
- `RoofFeature` contains the complete GeoJSON `Polygon`/`MultiPolygon` from
  the feature endpoint, the official `properties`, the endpoint URL, and the
  source timestamp selected from `datum_aenderung`/`datum_erstellung`. Its
  `area_m2`, `tilt_deg`, and `azimuth_deg` properties expose the validated
  physical area, roof tilt, and PV-library azimuth conversion directly.
- `OrthophotoResult.image_bytes` contains the JPEG returned by WMS. The result
  reports the exact `(x_m_per_pixel, y_m_per_pixel)` `gsd_m`, the LV95 BBOX,
  `north_up=True`, and SWISSIMAGE cache-update date when available. Decode it
  with `result.open_pil()`.
- `fetch_weather` returns `(DataFrame, metadata)`. The frame retains PVGIS
  fields and adds physics-ready aliases: `ghi`, `dni`, `dhi`, `temp_air`, and
  `wind_speed`. Its timezone-aware UTC `DatetimeIndex` uses non-leap reference
  year 2001; the original PVGIS timestamps remain in
  `source_timestamp_utc`. Thus the month-selected source years are never
  mistaken for a continuous multi-year simulation. Raw PVGIS columns are
  preserved; bounded irradiance/wind aliases are numeric and tiny negative
  upstream wind artefacts are clipped to zero for pvlib's non-negative input
  contract, with the exact per-column clipped counts recorded in metadata.
  PVGIS's `irradiance_time_offset` is also recorded and applied to the
  reference index before pvlib calculates solar position (the Brugg response
  reports `0.1789` hours).
- `fetch_horizon` returns `(profile, metadata)`, where each profile point is
  `{"A": azimuth_degrees, "H_hor": horizon_height_degrees}`.

## GeoAdmin address and Sonnendach roofs

Address search uses the official [GeoAdmin SearchServer](https://docs.geo.admin.ch/access-data/search.html)
endpoint:

```text
https://api3.geo.admin.ch/rest/services/ech/SearchServer
```

The connector explicitly sends `type=locations`, `geometryFormat=geojson`,
and `sr=4326`. GeoAdmin SearchServer can expose surprising LV95 GeoJSON axis
orders, so all roof work derives LV95 with `pyproj` (`EPSG:4326` to
`EPSG:2056`, `always_xy=True`) rather than guessing from a returned array.

Roofs use the official [SFOE Sonnendach GeoAdmin example](https://github.com/SFOE/ApiDocumentation/blob/master/GeoAdminAPI_ExampleSonnendach.md)
and layer `ch.bfe.solarenergie-eignung-daecher`. A point `identify` call finds
feature IDs, then each ID is fetched through:

```text
https://api3.geo.admin.ch/rest/services/ech/MapServer/
  ch.bfe.solarenergie-eignung-daecher/{feature_id}
```

with `returnGeometry=true`, `geometryFormat=geojson`, and `sr=2056`. The API
therefore returns the full polygon in LV95, not a screen crop. An empty
`results` list is a valid “no roof at this point” result; HTTP or schema
failures raise `SourceError`.

`fetch_roofs(tolerance_m=...)` accepts a ground radius. GeoAdmin's identify
`tolerance` is a screen-pixel radius, so the connector converts metres to
pixels against its 256-pixel identify extent instead of passing metres as
pixels.

The official [BFE data-model documentation (version 1.5, 31 Jan 2023)](https://pubdb.bfe.admin.ch/de/publication/download/9665)
defines the roof attributes used here:

- `FLAECHE` (`flaeche`) is the physical, inclined roof area in **m²**, which
  is the maximum module area before exclusions.
- `NEIGUNG` (`neigung`) is the roof angle to the horizontal in **degrees**;
  `0` is horizontal.
- `AUSRICHTUNG` (`ausrichtung`) is a signed compass convention centred on
  south: south `0°`, east `-90°`, west `+90°`, and north `-180°`/`180°`.
  It must be converted explicitly before passing it to a library expecting a
  north-zero azimuth convention; `RoofFeature.azimuth_deg` performs that
  conversion.

## SWISSIMAGE orthophoto

`fetch_orthophoto` uses the official [GeoAdmin WMS 1.3 documentation](https://docs.geo.admin.ch/visualize-data/wms.html)
and current-product layer `ch.swisstopo.swissimage-product`:

```text
https://wms.geo.admin.ch/
```

The request is `GetMap`, `CRS=EPSG:2056`, `FORMAT=image/jpeg`, and BBOX order
`min_easting,min_northing,max_easting,max_northing`. WMS returns row 0 at the
northern (`max_northing`) edge, so the result is north-up. Ground sampling is
reported rather than inferred from a hidden zoom:

```text
gsd_x = (max_easting - min_easting) / width
gsd_y = (max_northing - min_northing) / height
```

For provenance the connector also queries the official cache-update endpoint
for the product:

```text
https://api3.geo.admin.ch/rest/services/ech/MapServer/
  ch.swisstopo.swissimage-product/cacheUpdate
```

The returned `cache_update` value is the `source_data_date` when the endpoint
is available. The image remains usable if that optional metadata request is
temporarily unavailable; `source_data_date` is then explicitly `None`.

## PVGIS weather and optional horizon

Weather uses the [European Commission JRC PVGIS 5.3 non-interactive API](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en)
and its [TMY output definition](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en):

```text
https://re.jrc.ec.europa.eu/api/v5_3/tmy
```

The query requests JSON, UTC timestamps, and the DEM horizon (`usehorizon=1`).
The raw response is cached under `data/cache/` by default in a deterministic
SHA-256-named envelope containing the exact request, source URL, retrieval
time, and untouched upstream payload. Pass `refresh=True` to re-query PVGIS;
otherwise an existing cache entry is used. The metadata reports the PVGIS
radiation and meteo databases and their available source period (for example,
`PVGIS-SARAH3`, `ERA5`, `2005-2023`).

The optional `fetch_horizon` call uses:

```text
https://re.jrc.ec.europa.eu/api/v5_3/printhorizon
```

The connector preserves the raw PVGIS `A` convention: -180° north, -90° east,
0° south and +90° west. The physics module converts these rows to the internal
north-zero clockwise convention. Use it only if the simulation config applies this horizon;
PVGIS TMY already requests its DEM horizon for the selected meteorological
data.

## Brugg live smoke sample (10 September 2026)

The connectors were executed against real services for `Hauptstrasse 1, 5200
Brugg` (WGS84 `47.4831161499, 8.2071113586`):

- `data/cache/brugg_geocode.json`: GeoAdmin result and derived LV95
  `2657921.2338, 1259433.3161`.
- `data/cache/brugg_roofs.json`: one Sonnendach roof, feature `12631121`,
  `flaeche=1706.83844499 m²`, `ausrichtung=-180°`, `neigung=0°`, and full
  MultiPolygon geometry.
- `data/cache/brugg_orthophoto.jpg`: 512 x 512 JPEG from SWISSIMAGE over the
  selected roof extent (79,405 bytes in the smoke run); the connector reports
  `gsd_m=(0.1873047, 0.1445313)` and source cache date
  `2026-08-25T19:27:03`.
- `data/cache/pvgis_tmy_v5_3_aa98d24d5e385514c841.json`: raw PVGIS envelope;
  8,760 hourly rows, source period `2005-2023`, `PVGIS-SARAH3` radiation and
  `ERA5` meteorology. The earlier v5.2 cache remains locally available for
  reproducibility under `data/cache/pvgis_tmy_v5_2_aa98d24d5e385514c841.json`.
- `data/cache/brugg_weather_metadata.json`: the connector provenance record;
  `data/cache/brugg_horizon.json`: the optional 49-point DEM horizon profile.

These samples are local verification artifacts, not substitutes for a live
request. Their dates and bytes are recorded so a reviewer can distinguish
local cache state from current upstream state.
