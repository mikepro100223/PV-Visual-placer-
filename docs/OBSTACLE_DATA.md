# RID2 rooftop-obstacle data

The obstacle model uses the public [RID2 / Roof Information Dataset 2
Zenodo record](https://zenodo.org/records/14062580), released under CC BY 4.0.
The official [Zenodo API record](https://zenodo.org/api/records/14062580) is the
provenance authority for the archive and its checksum.

## Acquired source

- Archive: `data/raw/rid2/roof_information_dataset_2.zip`
- Direct endpoint: `https://zenodo.org/api/records/14062580/files/roof_information_dataset_2.zip/content`
- Expected size: 6,957,075,811 bytes
- Expected MD5: `9683e12c911358528416849c0074e17d`
- Coordinate reference system: EPSG:28992 (Amersfoort / RD New)
- Provenance sidecar: `data/raw/rid2/source-metadata.json`

`ensure_rid2_archive()` verifies the exact byte count and MD5 before a local
archive is used. The ZIP is retained unchanged; derived data is written below
`data/processed/rid2`.

The converter also records `archive_verified` in `splits.json` and refuses to
delete or overwrite an existing output path. A structurally similar test
archive is never given RID2's license claim.

## Conversion contract

RID2 supplies 1,819 roof-centred 512x512 PNG images plus a GeoJSON footprint
file and global superstructure GeoJSON. The converter intersects each
superstructure geometry with each EPSG:28992 image footprint, maps world
coordinates to image pixels (north-up, with the GeoTIFF top-left tie point),
and emits normalized YOLO segmentation polygons.

Only those PNGs are copied into the processed tree; the redundant GeoTIFF
imagery and source masks remain in the original archive and are not duplicated.

The current conversion produced 1,819 images and 58,342 polygons (813 MB):

All 1,819 image footprints were checked against the supplied GeoTIFF tiepoint
and pixel-scale metadata: 512 × 512 pixels at 0.08 m, covering 40.96 m squares,
with no transform mismatches. The Ultralytics scanner removes 106 duplicate
training rows (all balcony), leaving 58,236 effective rows across all splits.
Sixty duplicates originate in repeated source geometry; another 46 collapse
after clipping and segment-to-box conversion. Validation/test have no scanner
duplicates. The original derived labels stay unchanged; the audit is saved in
`artifacts/data-review/rid2-duplicate-audit.json`.

| Canonical class | Polygons |
| --- | ---: |
| solar_panel | 6,724 |
| chimney | 4,593 |
| skylight | 3,622 |
| dormer | 6,152 |
| roof_window | 5,572 |
| hvac | 2,201 |
| tv_dish | 167 |
| ladder | 281 |
| balcony | 1,780 |
| wall | 60 |
| other | 27,190 |

`PVModule` maps to `solar_panel`; `AC Outlet` and `AC System` map to `hvac`;
`Window` remains `roof_window` and is not silently merged with `Skylight`.
The remaining raw labels retain their explicit meaning in
`data/processed/rid2/splits.json`.

The dataset layout and class names are in
[`data/processed/rid2/dataset.yaml`](../data/processed/rid2/dataset.yaml),
with `images/{train,val,test}` and matching `labels/{train,val,test}`.

## Split and label limitations

The roof-centred case study has no matching official train/test CSV entries,
so the converter uses a deterministic spatial split over four 10 km Dutch
RD- New groups:

- train: 1,035 images (`09-43`, `23-44`)
- validation: 680 images (`13-44`)
- test: 104 images (`09-38`)

No spatial group occurs in more than one split. This avoids near-duplicate
neighbouring aerial tiles leaking across evaluation boundaries.

The held-out test region contains no `ladder`, `balcony`, or `wall` polygons
(and only 3 `tv_dish` and 22 `hvac` polygons). Those rare-class gaps are a
property of the geographic holdout, not filled with synthetic or leaked
examples; use the train/validation counts in `splits.json` when interpreting
per-class metrics.

RID2 explicitly does **not** label tree overhangs or shadows. They are listed
as `unsupported_not_negative` in `splits.json`; they must not be interpreted
as background negatives. The 132 published polyline features are also
recorded and skipped as area labels rather than silently rasterized.

The Swiss single-class solar-panel source remains separate. Do not append it
to this multiclass tree as negative obstacle examples: an absent obstacle
label is not evidence that a Swiss roof is obstacle-free.

For georeferenced inference the dashboard uses overlapping 40.96 m crops to
retain the RID2 training context, rather than shrinking a whole 100 m scene
to one 640-pixel input. Crop predictions return to original-image coordinates;
area accounting unions their overlaps. Polygon counts are not object counts.
This scale correction is not a validation of transfer to Swiss imagery. On
unreferenced uploads the geographic scale is unknown; inference remains a visual
diagnostic and cannot yield metric exclusions.

## Rebuild

From the repository root (after the archive has been acquired):

```bash
uv run python - <<'PY'
from pathlib import Path
from rooftop_pv.obstacle_data import ensure_rid2_archive, fetch_rid2_metadata, prepare_rid2_dataset

raw = Path("data/raw/rid2")
archive = ensure_rid2_archive(raw)
fetch_rid2_metadata(raw)
prepare_rid2_dataset(archive, Path("data/processed/rid2"), seed=42)
PY
```

For a deterministic small local smoke dataset, pass `max_images=...` (at
least three and including at least three spatial groups). Do not train during
conversion; training consumes the generated YAML only after the manifest and
class-count checks have been reviewed.
