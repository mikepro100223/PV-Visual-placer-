# Project and evaluation contract

## Purpose

Estimate additional installable rooftop PV capacity in Switzerland. Segment existing
PV installations from aerial imagery with **YOLO11 instance segmentation**, align
them to Sonnendach roof planes, subtract occupied space, and estimate hourly/monthly
solar electricity from roof orientation, slope and weather. Intended user: the
hackathon team and people comparing roofs for further assessment.

## Scope and assumptions

- First model: the official challenge's labeled Swiss PV dataset. Only label classes
  actually present in the dataset may be advertised as trained capabilities.
- Obstacles must be supplied as verified geometry or trained from a separately
  documented labeled dataset. An absence of obstacle labels is not proof of a clear roof.
- A single aerial RGB image cannot reliably establish roof tilt, load capacity,
  installation safety, wiring layout or year-round tree/building shade. Roof geometry
  comes from Sonnendach or explicit manual input. Unknowns remain visible.
- Solar yield uses hourly irradiance, air temperature, wind and solar position via
  pvlib, with explicit system/loss assumptions. Typical-year weather estimates long-term
  yield; it is not tomorrow's forecast.
- No paid cloud compute is provisioned. Local Apple MPS or CPU is used.

## Data and prediction contract

Images are north-up orthophotos with documented source, acquisition metadata where
available, pixel dimensions and ground resolution. Pixel predictions are polygons in
the original image coordinate frame. Area computations use EPSG:2056 metres and take
the union of clipped exclusions before converting horizontal to sloped roof area.
Azimuth within the application is degrees clockwise from north (south = 180).

Source downloads, labels and generated weights stay in ignored local directories.
Dataset preparation records provenance, deterministic split membership and hashes.
Related scenes/sites and duplicates must not cross train/validation/test boundaries.
If geographic metadata is unavailable, the remaining spatial leakage risk must be
reported. Held-out test labels do not influence fitting or threshold selection.

## Evaluation and mistake budget

Primary segmentation measures: held-out mask mAP50-95 and union-mask IoU. Report mask
precision/recall, area error, positive/negative counts, confidence threshold and
false-positive/false-negative examples. Compare against the unadapted starting model
only where its class semantics allow comparison; COCO classes are not PV labels.
Simple empty-mask and full-mask baselines are valid for area/IoU comparisons.

Provisional hackathon model gates, declared before training: mask mAP50 >= 0.70,
mask mAP50-95 >= 0.45, held-out union IoU >= 0.65. Passing these gates establishes a
dataset benchmark only, not generalization to every Swiss roof or obstacle recognition.
Failing a gate leaves the model marked experimental and does not block diagnostic use.
For a multiclass PV-plus-obstacle model, union IoU covers all annotated classes;
the non-PV obstacle union must also reach 0.65 separately, so PV performance cannot
conceal missing obstacles. Unsupported classes remain unassessed rather than certified.
Unacceptable: presenting fabricated metrics, counting existing panels as free space,
claiming unsupported classes, or treating unavailable inputs as measured zero losses.

## First experiment and fallback

1. Inventory/download/validate actual data and create immutable split manifests.
2. Verify YOLO11 segmentation can train on real labels with a short local baseline.
3. Train a bounded reproducible run, save best/last checkpoints and evaluate held-out
   examples. Use observed runtime to select a practical full training configuration.
4. Inspect failures and document metrics before calling a model ready.
5. Integrate inference with real geodata, PV simulation and a local interactive app.

Fallback is manual/verified exclusions and explicit physical scenarios. Any low-quality
model output is shown as experimental and editable; it never silently proves roof suitability.

## Acceptance evidence

Required deliverables: environment lock and requirements file; download/preparation,
training, evaluation and inference commands; geometry and physics tests; runnable local
dashboard; source/feature/output documentation; actual run status, measured metrics,
artifacts and limitations. No git commit or publication is part of this request.

Challenge: https://www.energydatahackdays.ch/challenges/ai-for-accurate-rooftop-pv-potential
