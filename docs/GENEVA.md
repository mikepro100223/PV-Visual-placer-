# Geneva roof superstructures

## Official source

This adapter uses the open SITG catalogue entry [Superstructures des toits des bâtiments](https://sitg.ge.ch/donnees/cad-batiment-horsol-toit-sp) and its official [ArcGIS FeatureServer](https://vector.sitg.ge.ch/arcgis/rest/services/CAD_BATIMENT_HORSOL_TOIT_SP/FeatureServer). Layer `0` is `CAD_BATIMENT_HORSOL_TOIT_SP`, an `esriGeometryPolygon` layer in CH1903+ / LV95 (`EPSG:2056`). The catalogue describes 2-D roof-superstructure footprints projected from 3-D building digitisation, with fields `OBJECTID`, `EGID`, `ALTITUDE_MIN`, `ALTITUDE_MAX`, and `DATE_LEVE`. Catalogue metadata says the layer is updated weekly and is open access; its usage restriction is currently listed as unspecified, so verify the current SITG conditions before redistribution.

The [STDL PROJ-ROOFTOPS ground-truth notes](https://tech.stdl.ch/PROJ-ROOFTOPS/#24-ground-truth) are an important boundary: their manually vectorized ground truth was produced on synchronized 2019 Geneva true orthophotos and LiDAR for 122 selected buildings. The SITG superstructure layer is a current, weekly-updated vector inventory and must not be assumed to be temporally aligned with the SwissImage training tiles or any other imagery.

## Bounded fetch

`rooftop_pv.geneva.fetch_superstructures` accepts one LV95 bbox and enforces a maximum 5 km side length and the official Geneva extent. It requests GeoJSON from layer `0`, asks for only the documented attributes, paginates at most 4,000 records per page, and stops at a 20,000-feature safety cap. The cache stores the raw GeoJSON plus query provenance and page offsets; no imagery is downloaded.

```python
from pathlib import Path

from rooftop_pv.geneva import fetch_superstructures, superstructure_exclusions

result = fetch_superstructures(
    (2_487_000, 1_111_000, 2_488_000, 1_112_000),
    cache_dir=Path("data/cache/geneva"),
)
obstacles = superstructure_exclusions(result)
```

The bounded live verification used that 1 km × 1 km bbox and returned 3 valid polygon features in one page. Its ignored cache is:

```text
data/cache/geneva/geneva-superstructures-4fd994ca69c60bbb56d6.json
```

`superstructure_exclusions` returns generic metric Shapely geometries for use as exclusions in `calculate_usable_area`; it does not invent chimney, skylight, HVAC, or other subtypes.

## Opt-in image labels

The adapter includes `geojson_to_yolo` for one explicitly georeferenced image at a time. The caller must provide:

- the image bounds in LV95 (`xmin, ymin, xmax, ymax`),
- pixel dimensions,
- an explicit acquisition date, and
- `alignment_status="verified"` with a non-empty human-readable alignment note.

The converter clips footprints to the image bounds, maps north-up LV95 coordinates to normalized YOLO polygons, and writes a sidecar containing feature IDs, bounds, date, CRS, and the alignment evidence. `write_yolo_annotation` refuses to create an empty `.txt`; an empty query result is not a negative because this source is not an exhaustive image annotation. Do not bulk-convert Geneva features into the frozen Swiss PV dataset.

Example for aligned 2019 true-orthophoto material only:

```python
from rooftop_pv.geneva import GenevaImagery, geojson_to_yolo, write_yolo_annotation

imagery = GenevaImagery(
    image_id="geneva-2019-tile.tif",
    bounds2056=(2_487_000, 1_110_900, 2_487_100, 1_111_100),
    size_px=(1500, 2941),
    acquired_at="2019-05-01",
)
artifact = geojson_to_yolo(
    result,
    imagery,
    alignment_status="verified",
    alignment_note="Synchronized 2019 true orthophoto and vector survey confirmed by source owner.",
)
write_yolo_annotation(artifact, Path("labels/geneva-2019-tile.txt"))
```

The converter is deliberately not wired into YOLO training. A future integration must first prove imagery/vector temporal and spatial alignment (especially the roof displacement difference between true orthophotos and ordinary orthophotos), define a site-level split, and keep unlabeled images out of the negative class.
