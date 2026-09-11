# SolarFit model update ? 10 September 2026

The deployed `rooftop_best.pt` is now the validation-selected **35-epoch PV fine-tune** (YOLO11n-seg, 640 px, batch 8, AdamW initial learning rate 0.001, seed 42). It starts from the 8 September checkpoint and retains the original geographic train/validation/test assignments. Training took 1,562 seconds. Checkpoints were selected on validation; the test set did not select the model or threshold.

| Pixel metric at confidence 0.25 | Previous PV model | Deployed PV model |
|---|---:|---:|
| Validation IoU (63 chips) | 0.7997 | 0.8538 |
| Test IoU (172 chips) | 0.5605 | 0.6087 |
| Test F1 | 0.7184 | 0.7568 |
| Test precision | 0.6179 | 0.6953 |
| Test recall | 0.8580 | 0.8302 |

This is a precision/recall trade-off, not a claim that every installation is found. Comparisons use the same 640-pixel input, 0.25 threshold and retina masks. Ground truth is converted polygon masks; the test set is the existing reused benchmark, not a newly blind national survey. Application heuristics, multi-view inference and roof placement are not included in these pixel scores. Exact checkpoint hashes, counts and per-image results are in `pv-retrain-validation.json` and `pv-retrain-test.json`.

## PV inference update, 11 September 2026

The application now runs the deployed checkpoint on the original and 180-degree
views, including the existing bounded overlapping tiles. Confidence remains
0.25. Additional masks must overlap original PV evidence over at least 25% of
their area; augmentation cannot introduce wholly unsupported arrays. Coordinates
are restored before merging, and merged courtyard holes are preserved by
partitioning the exterior-only API representation. This doubles inference views;
image-result caching remains enabled. No new checkpoint training was performed
for this inference update, and the single-pass metrics above do not measure it.

Reviewed live captures: Bachstrasse 15, Aarau recovers previously missed central
rows (98 to 66 proposed modules); Bleichemattstrasse 31 still excludes the courtyard
and both fan banks (169 proposed modules); Hirschlistrasse 3, Baden retains the
PV/awning distinction (12 proposed modules). Counts are planning outputs, not
ground-truth module capacity. `tests/test_pv_orientation.py` checks real Bachstrasse
PV interiors and non-PV controls, unsupported candidates and courtyard topology.
These cases do not establish flawless detection on other buildings.

## Geneva obstacle experiments

The provided SITG catalogue has 2D surveyed superstructure footprints, EGIDs, absolute elevations and survey dates, **not chimney/window subtype labels or exhaustive occupancy**. Prepared 600 georeferenced 64 m SWISSIMAGE chips at 10 cm/pixel: 409 train / 59 validation / 132 test, containing 5,816 / 496 / 2,351 polygon labels. Splits use 512 m blocks, discarded boundary chips and removal of cross-split EGIDs. Source and split records are included.

Two real models were trained locally:

- YOLO11n-seg (early-stopped after 33 epochs): no accepted validation detections at confidence 0.25. Rejected for deployment.
- Full-resolution U-Net (30 epochs): validation IoU 0.0489; test IoU 0.0632, precision 0.0731, recall 0.3177, using a validation-selected threshold of 0.5. **Rejected for automatic deployment.** `geneva_obstacle_research.pt` is a reproducible research checkpoint, not the application's `obstacle_best.pt`.

Do not interpret these low scores as an accurate all-obstacle detector. The sampled overlays show cadastral/image displacement; survey dates and imagery dates can differ, and missing labels treat real unrecorded objects as background. Better aligned, manually reviewed annotations are necessary before claiming image-based obstacle accuracy. The app continues to use surveyed Geneva footprints, DSM roof residuals and labelled image heuristics; compact two-cell raised vents are now preserved, while isolated noisy cells and thin neighbouring roof edges remain rejected.

`geneva-roof-evaluation.json` and its GeoJSON map 105 complete Sonnendach faces on 12 held-out Geneva chips into surface-metre usable areas after surveyed obstacles, detected PV and clearances. This is a georeferenced reference calculation before shade/structural checks, **not an independent usable-area accuracy score**.

## Reproduction

```
python -m training.train_yolo --model .cache/rooftop-before-retraining.pt --epochs 35 --imgsz 640 --batch 8 --name pv_retrain
python -m training.prepare_geneva --output data/geneva_clean
python -m training.train_yolo --data data/geneva_clean/dataset.yaml --model yolo11n-seg.pt --epochs 40 --imgsz 640 --batch 8 --name geneva_obstacles
python -m training.train_obstacles --data data/geneva_clean
python -m training.evaluate_obstacles
python -m training.evaluate_geneva_roofs --obstacle-source survey
```

The previous PV checkpoint remains in Git history and locally at `.cache/rooftop-before-retraining.pt`. Restore it there to reproduce the fine-tune. Survey services can change; the manifest and source checksum identify this run. Research checkpoints contain only a tensor state dictionary and primitive metadata and are loaded with `weights_only=True`.

Sources: [Swiss PV dataset](https://www.kaggle.com/datasets/jeanprbt/swiss-solar-panels-segmentation), [SITG superstructures](https://sitg.ge.ch/donnees/cad-batiment-horsol-toit-sp), [STDL label limitations](https://tech.stdl.ch/PROJ-ROOFTOPS/#24-ground-truth), [SWISSIMAGE](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10), [Sonnendach](https://opendata.swiss/de/dataset/eignung-von-hausdachern-fur-die-nutzung-von-sonnenenergie).

---

The following is the historical 8 September baseline card, retained for provenance; it does not describe the newly deployed checkpoint.

# SolarFit Swiss PV baseline

**Model:** YOLO11n-seg, approximately 2.84 million parameters. Checkpoint: `rooftop_best.pt` (~6 MB). Trained locally on 8 September 2026 with an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM), PyTorch 2.10.0+cu128 and Ultralytics 8.4.144.

## Intended use and scope

Hackathon demonstration and research on segmenting existing PV **installation regions** in top-down Swiss aerial imagery. This model has one class: `0: existing_pv`. It does **not** detect chimneys, skylights, dormers, roof boundaries or general obstacles. Supply those as manual exclusions in SolarFit until a fully annotated model is available. A PV region can contain many modules.

## Data and training

- Source: Jean Perbet’s [Swiss solar panels segmentation dataset](https://www.kaggle.com/datasets/jeanprbt/swiss-solar-panels-segmentation), version 1, Kaggle-listed CC0. Imagery: SWISSIMAGE / swisstopo, 10 cm/pixel.
- 758 source image/mask pairs. Grouped by integer kilometre location across capture years before deterministic splitting. See `split-manifest.json` for each source assignment.
- 579 train / 59 validation / 120 test source images; conversion yields 915 / 63 / 172 tiles respectively after 512-pixel tiling and negative sampling.
- Binary masks converted to external connected-component polygons; components under 12 pixels dropped. Holes are not represented by the conversion. Region boundaries at tile edges may be truncated.
- Initial weights: Ultralytics `yolo11n-seg.pt`. Trained 12 epochs, image size 512, batch 8, AMP, 2 data-loader workers, seed 42, rotations ±30°, horizontal/vertical flips, default YOLO augmentation and optimiser. Best validation checkpoint installed.
- Training time recorded in history: about 293 seconds, excluding environment downloads, initialisation and final evaluation. Training allocated approximately 1.12 GB GPU memory in the progress report; total desktop/GPU usage is higher.

Reproduce with `training/prepare_dataset.py` then `training/train_yolo.py --model yolo11n-seg.pt --imgsz 512 --batch 8 --epochs 12 --name solarfit_baseline --install`. Hardware/library differences may change results.

## Held-out test evaluation

172 test tiles, evaluated at 512 px, batch 4, on the RTX 4070. Exact values are in `evaluation.json`.

| Metric | Boxes | Segmentation masks |
|---|---:|---:|
| Precision | 0.653 | 0.659 |
| Recall | 0.728 | 0.731 |
| mAP50 | 0.730 | **0.736** |
| mAP50–95 | 0.543 | **0.528** |

Reported batched inference: **5.82 ms/image**, excluding preprocessing and postprocessing. This is not end-to-end upload latency. The application runs at 640 px by default and includes geometry/packing; measured HTTP timings for actual examples are in `benchmark.json`. First-use model/CUDA loading is materially slower than subsequent requests; layout-only recalculations reuse detections.

Validation mask mAP50 was approximately 0.710 and mask mAP50–95 0.487. The best checkpoint was selected on validation, not test performance. Test data was used for this final evaluation only. Example selection was for UI demonstration and is not independent accuracy evidence.

## Limitations

Additional pixel occupancy evaluation at the application's 640 px input and
0.25 confidence threshold is recorded in [pixel-evaluation.json](pixel-evaluation.json):
IoU 0.560, F1 0.718, precision 0.615, recall 0.861 across 172 test tiles.
These are micro-aggregated pixel metrics against converted polygon masks, which
omit holes and tiny components. They differ from the instance metrics above.
Kilometre groups have zero split overlap; adjacent groups can still cross splits.
Reproduce with `python -m training.evaluate_pixels`.

Precision and recall show that both missed PV and false positives remain. The dataset is geographically narrow; the grouping reduces direct tile leakage but does not establish robustness across countries, sensors, seasons or screenshot styles. Heavy shadows, small panels, perspective, map overlays and screenshot rescaling can reduce performance. Test mask performance does not prove correct additional-panel counts.

No obstacle, roof-plane, usable-area or installable-capacity ground truth was supplied. **Capacity MAE and obstacle accuracy are unavailable**, not zero. Confidence values are uncalibrated detector scores. Roof structural suitability, pitch, access and regulatory clearances require separate review.

Ultralytics model/software licensing applies (AGPL-3.0 or commercial terms). Load only trusted checkpoint files.
