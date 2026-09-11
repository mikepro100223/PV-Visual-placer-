# Object detection improvement: September 2026

## Iteration compact

**User and decision:** A planner clicks a Swiss roof and needs existing PV and
physical rooftop obstacles separated consistently so the proposed layout does
not cover installed modules or waste clear roof area. The decision is whether a
candidate is safe enough to replace the current RID checkpoint.

**Original baseline:** commit `b7314a7`, `models/swiss_best.pt` SHA-256
`086c85ef...3aa83e2`, and `models/rid_best.pt` SHA-256
`c9be26fb...6e3e044`. The published Swiss held-out mask metrics are precision
0.842, recall 0.766, mAP50 0.865 and mAP50-95 0.594. Published RID mask metrics
are precision 0.524, recall 0.466, mAP50 0.469 and mAP50-95 0.226.

**Observed failure:** on the supplied complex flat-roof case (`building_id`
2623385), 91.17 m² was simultaneously labelled as PV and a hard obstacle. The
UI also treated the semantic class `shadow` as though it were a physical roof
object. This created overlapping blue/orange masks and conservative empty
areas. Some large height-derived polygons represent real roof superstructures,
so they must not be removed merely to simplify the overlay.

**Acceptable errors:** a small uncertain model mask may be omitted when measured
height or survey evidence is unavailable, provided the UI keeps the result
provisional. Missing a physical object, overwriting measured evidence, or
placing new panels over confidently detected existing PV is unacceptable.

## Data contract

| Dataset | Use | Split integrity | Classes |
| --- | --- | --- | --- |
| Swiss PV, 758 images | Existing-PV regression only | 608 train / 80 val / 70 sealed test, source-group split | PV |
| RID2, 1,819 images | Physical-object domain adaptation | 1,035 train / 680 val / 104 sealed test, geographic groups | 11 source classes |
| Supplied Swiss roofs | End-to-end regression cases | Never used as labelled test truth | Visual/system consistency |

RID2 is converted into a separate derived dataset by
`scripts/prepare_rid2_yucan.py`. The original files and split lists are not
modified. Mapping is explicit: solar panel to PV; roof window to skylight;
HVAC, dish, balcony, wall and other to unknown obstacle. RID2 has no shadow or
tree labels, so none are invented. Exact polygons which become duplicates when
several source classes collapse into `unknown_obstacle` are removed. The final
derived set contains 1,819 images and 58,270 polygons. This limitation must be
considered before a fine-tuned checkpoint is promoted.

## Baseline and gates

Yucan's RID checkpoint evaluated on the mapped 680-image RID2 validation split
at 1024 px with mask mAP50 0.166 and mAP50-95 0.090. Per-class mask mAP50 was
0.274 PV, 0.069 dormer, 0.306 skylight, 0.001 ladder, 0.319 chimney and 0.024
unknown obstacle. These cross-domain numbers are deliberately separate from
the original RID held-out metrics.

A replacement checkpoint may be selected only when all applicable gates pass:

1. Validation mask mAP50 improves by at least 20% relative on mapped RID2, with
   gains coming from physical obstacle classes rather than PV alone.
2. The 104-image RID2 test split is opened once, after candidate selection, and
   confirms the validation direction.
3. `scripts/evaluate_hard_cases.py` passes against the published obstacle model
   under the same specialist ownership rule: existing Swiss-PV area stays
   within 1 m², PV/hard-obstacle conflict falls at least 65%, the complex-roof
   layout changes by no more than five modules, and shadows remain advisory.
4. All unit/integration tests, frontend type-check/lint/build, and API smoke
   checks pass.
5. The original checkpoints remain available for immediate rollback. No
   candidate is copied into `models/` before the gates pass.

## Implemented system correction

When the dedicated Swiss PV checkpoint is available it owns the PV class; RID
is used for PV only as a fallback if the Swiss checkpoint is missing. Pure RID
obstacle pixels are removed where they conflict with Swiss PV at confidence
0.5 or higher. Height, rooflight and Geneva survey evidence is never removed by
this rule. Same-class tiled masks continue to be stitched so large
installations retain full coverage, while merged provenance is deterministic.
Shadows remain visible as dashed advisory regions but no longer block module
placement.

## Promoted candidate

The candidate fine-tuned the published Yucan YOLO11s-seg checkpoint on mapped
RID2. It completed 14 of 15 requested epochs before patience-based early
stopping. Training used Apple MPS, seed 42, AdamW, cosine decay, conservative
color/scale augmentation and `mask_ratio=2`. PyTorch warns that one MPS
accumulation kernel is not bitwise deterministic even with deterministic mode.

| 1024 px mapped RID2 | Baseline mask mAP50 | Candidate mask mAP50 | Baseline mAP50-95 | Candidate mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Validation, 680 images | 0.166 | **0.417** | 0.090 | **0.222** |
| Sealed test, 104 images | 0.232 | **0.326** | 0.134 | **0.188** |

On validation, mean mAP50 over dormer, skylight, ladder, chimney and unknown
obstacle improved from 0.144 to 0.385. On the sealed test split every
represented physical obstacle class improved. The candidate's RID PV side
class produced false positives on a supplied roof; specialist ownership keeps
those out of production and retains the Swiss model as the only PV source.

With obstacle confidence calibrated to 0.30, building 2623385 retained exactly
462.35 m² of Swiss-PV, reduced PV/hard-obstacle overlap from 19.82 to 3.30 m²
(83.4%), and changed the layout only from 411 to 409 modules. All four
repeatable system gates passed. The promoted checkpoint SHA-256 is
`ac8d2e68698f48140f8407628bb45269c39eeb95a22deaa89f1ebd95e8279a93`.
The original checkpoint remains at
`artifacts/checkpoints/rid_best_yucan_b7314a7_c9be26fb.pt` for rollback.

RID2 provides no tree or shadow supervision. Those two head classes are
therefore not claimed as improved; measured obstacles remain authoritative and
image shadows are advisory rather than physical exclusions.

## Follow-up on supplied roof references: 11 September 2026

The production checkpoint was continued at 1024 px with a small AdamW learning
rate, stronger small-mask resolution and conservative augmentation. Training
was stopped after epoch 12 when the validation curve plateaued. The best
checkpoint improved mapped RID2 validation mask mAP50 from 0.417 to 0.507 and
mAP50-95 from 0.222 to 0.276. Per-class mask mAP50 reached 0.685 PV, 0.432
dormer, 0.714 skylight, 0.501 ladder, 0.601 chimney and 0.111 unknown obstacle.

That checkpoint was not promoted. At the production confidence of 0.30 the
six-roof system gate regressed building 2623385 from 3.30 to 20.06 m² of
PV/hard-obstacle overlap and changed the layout from 409 to 404 modules. The
main failure was PV texture classified by RID as skylight. Raising the global
obstacle threshold reduced the overlap, but caused an unacceptable layout jump
of 452 modules or more. On the supplied PV reference, the Swiss specialist
still detected the installation at 0.917 confidence through the actual padded
app pipeline.

A second conservative experiment added only the 608 Swiss-PV training images,
kept both projects' validation and test splits out of training, froze the
backbone and used a 2e-5 learning rate. It removed large RID false obstacles on
the supplied PV crop, but after one epoch RID2 validation fell to 0.373 mask
mAP50 and 0.207 mAP50-95 and a real flat-roof skylight disappeared. The run was
stopped and rejected. The sealed RID2 test split was not reopened because
neither follow-up candidate passed validation plus system selection gates.

Reproducible outputs are under `artifacts/iteration-20260911/` and the training
directories named `rid-1024-small-object` and `rid-swiss-pv-hard-negative`.
Production remains SHA-256 `ac8d2e68...e8279a93`; the next useful data step is
fully annotating Swiss PV roofs for all visible RID classes, particularly hard
negatives where panels resemble skylights and roof seams resemble obstacles.

## Reproduction

```bash
.venv/bin/python scripts/prepare_rid2_yucan.py
.venv/bin/yolo segment train model=models/rid_best.pt \
  data=data/processed/rid2-yucan/dataset.yaml epochs=15 imgsz=640 \
  batch=4 device=mps workers=0 seed=42 deterministic=True patience=5 \
  optimizer=AdamW lr0=0.0002 lrf=0.01 cos_lr=True warmup_epochs=2 \
  degrees=180 flipud=0.5 fliplr=0.5 scale=0.2 translate=0.1 \
  hsv_h=0.015 hsv_s=0.3 hsv_v=0.3 mosaic=0.5 mixup=0.0 \
  close_mosaic=4 mask_ratio=2 overlap_mask=True cache=False amp=False
.venv/bin/yolo segment val model=models/rid_best.pt \
  data=data/processed/rid2-yucan/dataset.yaml split=val imgsz=1024 \
  batch=2 device=mps plots=False
.venv/bin/python scripts/evaluate_hard_cases.py \
  --output artifacts/hard-cases/candidate-final.json
```

Training arguments and curves for candidates live under `runs/segment/` and
are ignored by Git. Generated data and experiment artifacts are not committed.

## Final gated attempt: 11 September 2026

Status: **not trained, not validated, not tested, not promoted**.

The hard-negative curation inspected all 608 Swiss-PV training images. It
produced 80 measurable candidate signals and a manually reviewed queue of 16
images: 12 were accepted as useful future annotation cases and 4 were rejected
or deferred as ambiguous. The handoff explicitly reports
`training_ready: false`.

No derived segmentation training split was created. Swiss-PV labels provide
only PV polygons; complete ground-truth polygons for dormers, skylights,
ladders, chimneys and unknown obstacles are absent. Model prediction polygons
were not accepted as labels. Training time was therefore 0 minutes, no
candidate checkpoint exists, RID2 validation was not rerun, and the sealed
RID2 test split was not reopened.

The production RID baseline remains mask mAP50 0.417 and mask mAP50-95 0.222
on RID2 validation. Production remains SHA-256
`ac8d2e68698f48140f8407628bb45269c39eeb95a22deaa89f1ebd95e8279a93`;
Swiss remains SHA-256
`086c85ef1a8cc078241a56044fd60aeb0dd7e6c8bea06ba217821128c3aa83e2`.
No checkpoint was copied or published.

The next safe step is exhaustive six-class polygon annotation of the 12
accepted Swiss-PV train images followed by independent completeness QA,
especially for small real skylights and chimneys. RID2 supplies no tree or
shadow labels, so no improvement for those classes is claimed.
