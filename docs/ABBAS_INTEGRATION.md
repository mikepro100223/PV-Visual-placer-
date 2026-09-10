# Selective Abbas integration on Yucan

Base: `Yucan` at `3c6ac5b32dbfc2c625bb251a6c73611a03c01567`.
Reviewed and selected source: `Abbas` at `12335ac46d200755d7c514e280d71dbd0f2fc78a`.
No branch merge, cherry-pick, or replacement of Yucan's application was used.

## Analysis and selection

| Abbas source | Decision |
| --- | --- |
| `elevation_service.py` | Port DSM download, mosaic, per-facet robust plane fit and obstacle detection. Keep Yucan's official pitch/geometry for placement; do not import Abbas roof-model, sunlight or bridge orchestration. |
| `rooflight_service.py` | Port local colour/size-based skylight detection, excluding existing Yucan PV first. |
| `geneva_service.py` (new commit) | Port paginated surveyed roof-superstructure polygons, restricted to Geneva and checked for EPSG:2056. |
| `pv_segmentation.py` | Adapt full-image plus bounded crop inference for obstacles only, and same-class mask merging. Preserve repaired polygon parts and holes. |
| `yolo_service.py` (new commit) | Adopt obstacle-only separation and optional `OBSTACLE_MODEL_PATH`; reuse Yucan's single CUDA loader rather than adding another ML service. |
| PV checkpoint / PV colour detection | Do not import. The tracked Abbas checkpoint is PV-only, not an obstacle model. Yucan?s original Swiss + RID PV outputs are retained. Both checkpoints belong to Yucan; no Abbas PV detector is used. |
| SAM, optimizer, roof geometry, server, frontend | Do not import. Retain Yucan's working implementation. |

The copied obstacle modules are covered by the source AGPL license, preserved at
`app/obstacles/LICENSE`. Adaptations and integration code live in `app/`.

## Pipeline and contract

Existing `/api/status`, `/api/search`, `/api/analyze`, GeoJSON kinds and frontend controls remain compatible.
The analyze response adds `detections` and `obstacle_sources`.
Each detection contains `kind`, `confidence`, `polygon`, optional `holes`, `source`, `sources`, and explicit `crs: "EPSG:2056"`.
Coordinates are LV95 metres, not image pixels or latitude/longitude. GeoJSON overlays remain WGS84.
Image origin/scale and crop offsets are converted once at the inference boundary.

`existing_pv` and `pv` map to `pv_installation`; `unknown_obstacle` maps to `other_obstacle`.
Chimneys, skylights, dormers, ladders, trees and shadows remain distinct.
Different classes never merge into PV. Same-class duplicates retain the union and highest model confidence.
Measured/heuristic detections carry `confidence: null`: their values are not fabricated ML probabilities.

YOLO masks, DSM structures, image rooflights and available Geneva polygons are clipped to the selected roof,
normalised and passed together to Yucan `pack_building`. Its roof-plane metre transform, obstacle buffer,
edge setback, module dimensions, row spacing and shared occupancy ledger are unchanged.
Existing PV stays blue, obstacles orange and proposed panels green.

A PV regression was found after initially restricting Yucan to the Swiss checkpoint alone.
The original Yucan branch actually combines Swiss and RID PV outputs. This has been restored;
on identical Einstein-Haus imagery the restored PV union matches original Yucan exactly (63.4885 m?, zero symmetric difference).

## Models and dependencies

Default inference loads exactly three existing roles: Swiss PV, RID obstacles and roof segmentation.
RID native-crop PV predictions are retained at the PV confidence threshold. The Abbas full-image obstacle pass cannot contribute PV. Optional `OBSTACLE_MODEL_PATH=models/obstacle_best.pt` replaces the RID role;
it does not add a fourth model. The new Abbas branch does not include that checkpoint, so it is not assumed available.
Obstacle checkpoints must be segmentation models with supported rooftop classes, not generic COCO or PV-only weights.
CUDA/PyTorch and current Ultralytics version are retained. No SAM, DracoPy, second web framework or ML runtime added.
NumPy, OpenCV and HTTPX are now explicit dependencies; the existing local environment already provides them.

DSM tiles are cached beneath `data/cache/dsm` with bounded tile count and cache size.
Optional network sources have time budgets and failures generate warnings and a provisional layout;
they do not silently establish that a roof is clear.
The new integration does not launch training or replace PV weights.

## Validation and limits

Tests include copied Abbas DSM/rooflight cases plus combined-source placement, north-up coordinate conversion,
crop offsets, hole preservation, class aliases, deduplication, independent PV/obstacle confidence,
Geneva pagination/CRS failures, and graceful height-service failure.
Live Einstein-Haus analysis used real DSM and CUDA inference and returned 166 proposed modules;
new height and rooflight exclusions contribute to the result. This is a functional check, not accuracy ground truth.

DSM classification is based on height and footprint; `chimney` is a candidate class, not a verified object identity.
Colour-based skylights can miss non-blue windows or confuse reflections. Geneva survey polygons only cover Geneva
and may predate imagery. Dormer classification still comes from the existing trained obstacle model.
No new weather, structural, or regulatory suitability guarantees are introduced.
