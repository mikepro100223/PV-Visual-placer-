# Swiss PV segmentation data

## Source and acquisition

The training source is the public v1 release of [Jean Perbet's Swiss solar panels segmentation dataset](https://www.kaggle.com/datasets/jeanprbt/swiss-solar-panels-segmentation). The upstream project documents that the imagery is Swiss aerial imagery from swisstopo and that the task is binary pixel segmentation of photovoltaic installations: [jeanprbt/swiss-solar-panel-segmentation](https://github.com/jeanprbt/swiss-solar-panel-segmentation).

The acquisition is pinned to Kaggle `datasetVersionNumber=1` and does not use Kaggle credentials. The release metadata reports:

- dataset ref: `jeanprbt/swiss-solar-panels-segmentation`
- version: `1` (initial release, updated 2025-06-18)
- licence: `CC0: Public Domain`
- archive bytes: `333,905,642`
- archive SHA-256: `1cc078d5a7256acc492ac3e8c6a12ff3096a5c8309e9e33cdf58a27af6b49ce0`
- source members: 758 aerial JPEGs, 758 PV PNG masks, 758 roof-only JPEGs, 758 roof masks, `labels.json`, and four upstream weights

The downloaded archive and a credential-free metadata snapshot are kept under `data/raw/` (ignored by Git):

```text
data/raw/swiss-solar-panels-segmentation-v1.zip
data/raw/source-metadata.json
```

`source-metadata.json` preserves the public Kaggle metadata, licence, version, retrieval time, archive size, and digest. It contains no cookies, API tokens, or signed download URL.

## Preparation contract

Run from the repository root:

```bash
uv run python -m rooftop_pv.data download --raw-dir data/raw
uv run python -m rooftop_pv.data prepare \
  --raw-dir data/raw \
  --output-dir data/processed/swiss-pv \
  --seed 42
```

The prepared YOLO11 segmentation dataset is at:

```text
data/processed/swiss-pv/
├── dataset.yaml
├── splits.json
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── masks/{train,val,test}/*.png
```

`dataset.yaml` declares one and only one class: `solar_panel` (class id `0`). The original binary masks are retained under `masks/` so evaluation can compare predictions with the source raster and recover holes or connected-component semantics without depending on the polygon approximation.

The generated release currently contains 758 readable 1000 x 1000 JPEGs and 1,313 YOLO polygons:

| split | geographic groups | images | non-empty/total labels |
| --- | ---: | ---: | ---: |
| train | 61 | 608 | 315 / 608 |
| val | 8 | 80 | 39 / 80 |
| test | 7 | 70 | 34 / 70 |

The counts above are recorded in `splits.json`; masks and labels are matched by the same source stem. Empty label files are intentional negatives.

## Leakage and conversion safeguards

The source has no official train/validation/test split. `data.py` groups each tile by its integer SwissImage coordinate (`east-north`), excluding the imagery year and decimal 100 m subtile. The 76 geographic groups are assigned deterministically by SHA-256 ordering with seed `42`, so no neighbouring subtile/site group is shared across train, validation, and test. Image byte hashes were also checked for duplicate content across the release.

The converter thresholds the source grayscale masks, extracts external connected components, simplifies contours, and writes normalized YOLO polygons. It does not infer obstacles from `roofs/masks`; those are roof footprints and are not labels supported by this challenge. The prepared manifest records conversion fidelity: mean mask IoU `0.99584`, minimum IoU `0.80265`, and 387/758 masks reconstructed exactly. The exact source masks remain available for evaluation.

To prevent accidental deletion, `--overwrite` only clears a prior `swiss-pv` output identified by its own `splits.json` provenance marker; it refuses arbitrary or unrelated directories. ZIP members are path-validated before extraction and corrupt archives fail closed.
