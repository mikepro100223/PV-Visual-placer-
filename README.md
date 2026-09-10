# PV Visual Placer

Local Swiss aerial map, PyTorch YOLO11 segmentation and real-size solar-module placement. Click inside a house roof or search an address.

## Run

Double-click `start.cmd`, or run `.\start.ps1`, then open **http://127.0.0.1:5173**. The launcher runs the Python API on port 8000 and the map on port 5173. Logs are in `logs/`. Ctrl+C stops its servers, not training. Internet is needed for uncached Swiss imagery and roof lookup; no paid API key is needed.

Training and inference use CUDA-enabled PyTorch on the **RTX 4090 Laptop GPU (16 GB)**. Training refuses CPU execution. Set `PV_INFERENCE_DEVICE=cpu` only if you intentionally want CPU inference.

For a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Set-Location web
npm.cmd install
Set-Location ..
.\start.ps1
```

Tested with Python 3.14.3 and PyTorch 2.10.0+cu128. The frontend requires Node 22.13 or newer. The Sites starter is used locally; no deployment is needed.

## Detection and placement

- **Swiss PV model:** existing PV installation masks, including negative examples without panels. It detects coverage, not installation age or individual module electrical specifications.
- **Obstacle model:** RID classes PV, dormer, skylight, ladder, chimney, shadow, tree and other obstruction. These labels are separate from the PV-only dataset.
- **Roof model:** roof masks from the local dataset, supplemented by Geneva imagery paired with Sonnendach roof polygons. When available, placement requires agreement between detected roofs and Sonnendach geometry.
- **Unoccupied roof:** calculated as roof area minus detected PV and obstacles. Missing detections are not proof that an area is clear. The site visibly marks layouts provisional while a required model is unavailable.
- **Module:** Trina Vertex S+ TSM-NEG9RC.27, 450 W, **1.762 x 1.134 m**. There is no universal module size; this is one actual manufacturer model.
- **Spacing:** 0.60 m edge clearance, 0.50 m obstacle clearance, 0.10 m between modules, 0.35 m between pitched-roof rows, at least 1 m between flat-roof rows, and an 0.80 m access corridor on larger roof faces. Edge clearance and row spacing are adjustable. These are conservative prototype settings, not certified installation rules.
- **No stacking:** the building shares one occupancy ledger across all facets. Overlapping or duplicated roof polygons cannot receive another stack of panels. Complete modules must fit inside their roof face in physical roof-plane metres, accounting for pitch and aspect.

Annual energy uses Sonnendach radiation and an assumed 80% performance ratio. There is no new hourly weather, height-map shadow, structural or electrical-string simulation. Exact flat-roof rack tilt/spacing still requires installation design. Geodata and imagery can differ in date and alignment; small or obscured obstacles may be missed.

## Data from `datasets/`

The preparation uses `datasets/kaggle-swiss-solar-panels-segmentation` (or the flat `datasets/` structure if the nested directory is absent), `datasets/swissimage-geneva/*.tif`, and `datasets/SOLKAT_DACH.gpkg`.

| Dataset/model | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| PV masks | 518 | 150 | 90 |
| Roof masks plus Geneva roof crops | 905 | 156 | 106 |
| Reviewed RID obstacle masks | 911 | 282 | 154 |

The Swiss PV data contains 758 images: 388 positive and 370 negative. The two copies inside `datasets/` are not counted twice. Geneva contributes 413 additional roof crops from the local TIFFs and GeoPackage; one TIFF decoder failure is recorded in the audit. Geneva roof supervision is derived from geodata, not manually reviewed image labels. Overlapping crops and exact duplicates are removed between the relevant geographic splits.

**The Geneva obstacle CSV has heights/areas but no coordinates or polygons. It cannot train obstacle segmentation.** A georeferenced GeoJSON/GeoPackage/shapefile export would be needed to pair those obstacles with the aerial images. The already downloaded RID reviewed annotations provide obstacle supervision meanwhile.

Preparation reports, exact split membership, conversion quality, excluded files and mask overlays are in `data/reports/`. Training data and model weights are ignored by Git.

## Retrain on the GPU

```powershell
.\retrain.ps1
```

This prepares the local data and trains obstacles, roof boundaries, then PV sequentially on the GPU. It uses YOLO11s segmentation, pretrained initialization, up to 100 epochs per model, early stopping patience 20, image size 1024, seed 42, AdamW with cosine learning-rate decay, and orientation/lighting augmentation. Smaller obstacle batches and standard mask downsampling control VRAM use. It does not substitute bounding-box detection for segmentation.

To use the already audited prepared data:

```powershell
.\.venv\Scripts\python.exe scripts/retrain_local.py --prepared
```

To prepare without launching training:

```powershell
.\.venv\Scripts\python.exe scripts/retrain_local.py --prepare-only
```

For the original public downloads, use `scripts/download_data.py --swiss` and `scripts/download_rid.py`. Do not run overlapping training processes on a 16 GB GPU.

Outputs: `models/{swiss,rid,roof}_best.pt`, live `*_status.json`, final held-out `*_metrics.json`, and resumable checkpoints/curves under `runs/`. Check status files for actual completion; preparation alone is not training. RID scores evaluate the German source domain and do not establish Swiss obstacle accuracy.

The initial completed Swiss run (100 epochs, before the requested retraining) achieved held-out mask precision **0.842**, recall **0.766**, mAP50 **0.865** and mAP50-95 **0.594**. These are installation-region metrics, not a guarantee of roof capacity accuracy.

## Selective Abbas integration

Obstacle detection now combines Yucan RID with Abbas height-based structures, image rooflights and Geneva surveyed polygons. Yucan PV, roof geometry, usable-area calculation and placement remain in place. See [integration decisions, contract and validation](docs/ABBAS_INTEGRATION.md). An optional `OBSTACLE_MODEL_PATH` replaces the obstacle checkpoint without adding another model.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
Set-Location web
.\node_modules\.bin\tsc.cmd --noEmit
.\node_modules\.bin\oxlint.cmd app vite.config.ts
npm.cmd run build
```

Geometry tests cover physical dimensions at different pitches/aspects, roof holes and concavities, edge/obstacle clearances, overlapping facets, duplicate exclusions and reduced flat-roof density. Live API checks include an occupied sloped roof yielding no new panels and a flat house whose revised layout drops from 142 to 69 modules. The scoped lint command checks product code; the unused generated component catalog has upstream lint findings.

## Sources

- [Swiss PV dataset](https://www.kaggle.com/datasets/jeanprbt/swiss-solar-panels-segmentation), listed as CC0, and [authors' code](https://github.com/jeanprbt/swiss-solar-panel-segmentation).
- [RID dataset](https://mediatum.ub.tum.de/1655470), **CC BY-NC 4.0** for noncommercial use, and [authors' code](https://github.com/TUMFTM/RID). Credit Sebastian Krapf, Lukas Bogenrieder, Fabian Netzler, Georg Balke and Markus Lienkamp; consult the dataset README for imagery conditions.
- [SFOE Sonnendach API](https://github.com/SFOE/ApiDocumentation/blob/master/GeoAdminAPI_ExampleSonnendach.md), swisstopo imagery, and the local dataset providers' terms.
- [Trina module dimensions](https://www.trinasolar.com/en-glb/NEG9RC.27/) and [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11).

`TRAINING_PLAN.md` is the earlier broader proposal; this README describes the implemented prototype.
