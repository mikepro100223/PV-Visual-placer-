# PV Visual Placer: plan before training

Status: proposed design only. No dataset has been downloaded or audited, dependencies installed, model trained, or performance measured. The repository initially contained only a README. This plan uses YOLO11 instance segmentation with PyTorch through Ultralytics.

## What the model will train on

### Stage 1: existing photovoltaic installations

- Source: [Swiss solar panels segmentation on Kaggle](https://www.kaggle.com/datasets/jeanprbt/swiss-solar-panels-segmentation).
- The [dataset author's repository](https://github.com/jeanprbt/swiss-solar-panel-segmentation) describes approximately 750 Kaggle data points and a binary semantic segmentation task: solar-panel pixels versus background. This is an author-reported approximate count, not an audited count of usable image/mask pairs.
- Input: RGB aerial image crops. Supervision: corresponding PV masks converted to YOLO segmentation polygons. Initial foreground class: `pv_installation`.
- Treat contiguous annotated panel regions as installation regions. Binary masks do not supply individual module identities; disconnected components and touching installations need inspection during conversion. Do not claim individual panel counts from these labels.
- Inspect mask pixel values, holes, small objects, image alignment, acquisition dates, source resolution, and available geographic metadata before conversion. Compare polygon rasterizations against original masks to measure lost area, especially holes and gaps.
- Add independently reviewed swisstopo images with no PV installations and difficult negatives such as dark roof tiles, skylights, solar thermal collectors, water, and shadows. The quantity and geographic coverage depend on the data audit.

### Stage 2: rooftop obstacles

Create additional polygon annotations on representative swisstopo roof imagery for `chimney`, `skylight`, `dormer`, `roof_equipment`, and `solar_thermal_collector`, alongside `pv_installation`. These classes are proposed; the linked binary PV dataset does not establish annotations for them. Revise the taxonomy after checking whether image resolution supports reliable distinctions.

Human-reviewed SAM proposals can accelerate annotation. Unreviewed proposals are not ground truth. Use a separate obstacle model initially, or a unified model only on images exhaustively annotated for all its classes. Feeding PV-only annotations to a normal multiclass trainer would incorrectly teach it that unlabeled obstacles are background.

Use Sonnendach roof facets as geometry where available; roof segmentation is a separate annotation task if those facets are missing or outdated. An obstacle footprint blocks placement, while its height also determines a changing shadow. One cannot substitute for the other.

## Model and validation

Start from pretrained `yolo11s-seg.pt`, fine-tuned through the Ultralytics PyTorch implementation. The generic pretrained checkpoint is initialization; Swiss image/mask pairs provide the task-specific supervision. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11)

Proposed initial experiment: up to 100 epochs, early stopping with patience 20, seed 42, image size 1024 if native crop resolution warrants it, and a batch size selected after GPU memory inspection. These settings have not been benchmarked. Preserve source spatial detail and use overlapping inference tiles; resizing cannot recover unresolved obstacles.

Split approximately 70/15/15 into training/validation/test by geographic groups before tiling or augmentation. Keep the same building, overlapping coverage, and repeat acquisitions in one split, with separation between held-out geographic blocks. If locations cannot be recovered, document that a credible geographic generalization test is unavailable. Remove duplicate content. Apply rotations, flips, and moderate lighting changes only to training images, transforming masks identically.

Evaluate union-mask IoU/Dice, installation-region mask mAP, per-class precision/recall, small-obstacle recall, and blocked/usable roof area error in square metres. Converted binary components are proxy instances, so mAP is not evidence of individual-module detection. Use the test set only after choosing the model and thresholds on validation data. Report results by roof type, region, object size, and imagery quality, including uncertainty across roofs.

## How the full estimate considers other factors

YOLO learns visible object boundaries. The planned downstream geospatial and physical calculation combines those boundaries with the following inputs; none of these integrations is implemented yet.

| Factor | Input and treatment |
| --- | --- |
| Roof footprint, slope, aspect, and area | Sonnendach roof facets, checked against imagery and 3D roof geometry. Convert direction conventions explicitly. Calculate usable surface and module layout in each roof plane, not by counting map pixels alone. |
| Height map and local shading | swissSURFACE3D surface data for buildings/vegetation and suitable LiDAR or roof geometry for finer structures. Use sun-position-dependent shadows and sky visibility. Surface model resolution limits small-chimney reconstruction. |
| Terrain, altitude, mountain horizon | swissALTI3D terrain data for the surrounding horizon and elevation. A terrain model excludes buildings and trees, so it cannot by itself describe roof pitch or nearby obstructions. |
| Sun exposure | Latitude, longitude, timezone-aware timestamps, panel tilt and azimuth, seasonal solar position, incidence angle, direct/diffuse irradiance, and ground reflection. Evaluate each roof face separately. |
| Clouds, fog, seasonal weather | Historical hourly irradiance over representative years, with direct/diffuse components measured or estimated using documented methods. Use MeteoSwiss radiation/climate data where coverage permits; a current forecast does not estimate long-term annual production. Do not apply a second generic cloud penalty to already weather-adjusted radiation. |
| Temperature and wind | Weather time series, mounting configuration, and module temperature coefficients to model cell temperature and power changes. |
| Snow, dirt, and vegetation | Snow-cover and soiling loss scenarios using observations where available; distinguish snow losses from snow-enhanced albedo. Include seasonal foliage and changing surroundings where data support them. Mark missing measurements as assumptions. |
| Actual placement | Module dimensions, roof edges, ridges, obstacles, access paths, configurable setbacks, mounting tilt, flat-roof row spacing, and self-shading. Pack complete modules into valid polygons; area divided by panel area is only an upper bound. |
| Electrical performance | Module rating and technology, inverter efficiency and clipping, wiring, mismatch, partial shading, availability, and degradation. Use documented pvlib models with explicit loss assumptions; geometry alone does not resolve string-level shading losses. |
| Installation feasibility | Roof condition and load capacity, wind/snow design loads, mounting, heritage/planning restrictions, and grid connection require external records or assessment. Report these as unverified until supplied, never infer approval from aerial imagery. |
| Data quality | Preserve CRS, metre-per-pixel scale, timestamps, provenance, registration error, resolution, missing-data flags, and prediction confidence. Validate joins in Swiss LV95 (EPSG:2056), and transform coordinates for solar calculations. |

Sources: [Sonnendach](https://www.bfe.admin.ch/en/solar-roof), [swissSURFACE3D Raster](https://www.swisstopo.admin.ch/en/height-model-swisssurface3d-raster), [swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d), [MeteoSwiss climate data](https://opendatadocs.meteoswiss.ch/c-climate-data), [pvlib ModelChain](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.html).

## Intended outputs and accounting

1. Detected existing PV coverage and obstacle polygons with confidence and acquisition date.
2. Additional usable roof surface after subtracting the union of existing PV, obstacles, and buffered exclusion zones. Union overlapping exclusions to avoid double subtraction. For a planar facet, projected area divided by cos(slope) gives surface area, but placement still needs roof-plane geometry.
3. A feasible module layout and additional nameplate capacity: module count times module rated watts divided by 1000, in kWp.
4. Monthly and annual additional AC energy, in kWh, obtained by integrating modeled power over weather time steps. Report an uncertainty range and assumptions separately from measured inputs.
5. Existing and additional capacity reported separately. Imagery identifies occupied area; existing electrical capacity requires equipment data or an explicitly labeled estimate.

Audit which weather, shading, and losses Sonnendach's selected baseline already includes before applying corrections. Use its output as a comparison baseline or an explicitly accounted starting point; do not blindly apply the same losses twice. Compare usable-area estimates with reviewed roof layouts and, where available, energy estimates with metered installations and matching equipment/weather records.

## Required pre-training report

Before launching training, report the actual dataset version, paired-image count, foreground and negative counts, mask encoding, class counts, polygon-conversion quality, geographic split membership, leakage checks, sample overlays, annotation gaps, and available GPU/PyTorch environment. At present these are unknown because the files have not been obtained. This document describes the proposed training task, not a completed data audit or an operational capacity estimator.
