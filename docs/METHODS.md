# Method decisions and research context

## Vision and geometry are separate

The requested YOLO11 instance model learns occupied image regions. Sonnendach supplies
the roof plane and its slope/azimuth. Predictions are transformed using the actual
north-up image bounds, clipped to the selected roof, and unioned before subtraction.
Map areas are divided by cos(tilt) only once. Margins and packing are separate losses.

The [EPFL study by Castello et al. (2021)](https://doi.org/10.1088/1742-6596/2042/1/012002)
motivates combining segmentation with building geometry and evaluating suitable area.
Its segmentation task, data years, annotations and split differ from this project's;
its published scores are not benchmarks that our YOLO model can claim to match.

## Source-specific labels

The [Swiss PV reference project](https://github.com/jeanprbt/swiss-solar-panel-segmentation)
uses binary semantic masks and U-Net/DeepLabv3. We use its linked real dataset with
YOLO11 instead. Connected regions become YOLO polygons; the original masks remain
the reference for union IoU and area error. Polygon conversion itself is measured.

Swiss PV labels do not annotate every chimney, skylight or dormer. RID2 has a different
annotation scope and geography, so its obstacle detector is trained separately. Inference
can combine the supported classes while preserving their provenance and experimental status.

The [STDL rooftop study](https://tech.stdl.ch/PROJ-ROOFTOPS/#24-ground-truth)
explains why existing Geneva inventories are not complete image ground truth. It used
manually delineated objects on imagery aligned with the LiDAR acquisition. Consequently,
the SITG footprint layer is used here as a supplementary exclusion source. Conversion
to training labels requires an explicit check of image date and spatial alignment.
An unlisted structure is not automatically absent.

## Error analysis and uncertainty

The pinned Ultralytics 8.4.146 trainer selects `best.pt` and early stopping by
its native segmentation fitness: box mAP50-95 plus mask mAP50-95 on validation.
This is not necessarily the epoch with the highest mask mAP alone. The saved
checkpoint is then evaluated independently for the reported mask/area gates;
test performance is never used to choose an epoch.

The model evaluation records instance mask mAP, positive-image and aggregate occupied-mask
IoU, false-positive/false-negative pixels, and area error. The test split remains separate
from model fitting. Fixed operating confidence is stated in each report. Worst cases are
saved for review rather than discarded. There are no duplicate JPEG hashes in the initial
758-image Swiss preparation (checked 10 September 2026).

Roof-object boundaries, raster resolution and orthophoto displacement cause area
uncertainty. A WMS service refresh timestamp is not an aerial acquisition date. The
application must not interpret cache freshness as imagery-label temporal alignment.
Model scores measure the held-out dataset; they do not establish structural suitability
or exhaustive obstacle coverage for a new building.

During the unlabelled Brugg integration review, Ultralytics' merged contour export
was found to bridge disconnected predicted mask islands. Buffering those artificial
lines inflated obstacle setbacks. Inference now extracts separate positive-area
external contours directly from original-resolution raster masks. Degenerate contours
with no polygon area are skipped; class confidence is unchanged. Interior holes are
still filled by the exterior-only prediction representation, so these polygons are
not an exact raster vectorization. Original Swiss reference holes remain in evaluation.
Mask/component counts are not physical object counts. The area reports were recomputed
after this geometry fix with identical weights, confidence and inference size; no
test-label-driven parameter or training changes were made. New reports record hashes
of the evaluator and inference implementation in addition to the data and checkpoint.

## Weather and power

The independent pvlib model consumes a typical meteorological year, with source years
retained in provenance and calendar months joined into one non-leap simulation year.
Solar position uses the PVGIS irradiance timestamp offset. Irradiance, temperature,
wind, albedo and explicit losses determine hourly energy. Weather observations from a
particular upcoming year, local obstructions, module datasheets and engineering checks
are separate inputs, not outputs inferred by the image model.
