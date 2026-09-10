# Geneva image-to-roof validation

## Contract and fixed experiment

Goal: run the existing Swiss PV and RID2 obstacle checkpoints on real Geneva
SWISSIMAGE, join predictions to every Sonnendach roof plane intersecting a requested
LV95 area, and export an explicit usable-area candidate per roof ID.

The preceding turn made progress (two completed trainings, held-out reports,
tested Brugg integration), but did not prove Geneva inference or multi-roof output.
Those requirements remain open until the evidence below is generated and checked.

The user has not specified an exhaustive canton-wide boundary. The reusable command
will process every roof intersecting its explicit requested bbox, without a hidden
top-N subset. The live experiment uses three preselected 100 m query areas around:

1. Rue du Général-Dufour 24, Genève
2. Rue Gabrielle-Perret-Gentil 4, Genève
3. Route de Meyrin 49, Genève

All roofs intersecting each query area are targets, including crossing roof planes.
Imagery coverage is expanded to include their complete geometry, using bounded
100 m / 1000 pixel north-up SWISSIMAGE tiles. No clipped visible fragment is reported
as a complete rooftop. Source caps fail explicitly rather than silently dropping roofs.
Image-to-roof mapping uses actual EPSG:2056 bounds, not address proximity alone.

## Models and measurement

Checkpoints, confidence 0.25 and inference sizes (PV 512 / obstacle 640) are frozen.
No additional fitting, threshold search or test-label creation is part of this test.
RID2 uses 40.96 m contexts; PV uses 100 m contexts. Duplicated tile masks are unioned
for area; mask components are not physical object counts.

SITG is a dated, incomplete superstructure inventory, not exhaustive image truth.
Report spatial overlap as cross-source agreement only, never Geneva precision,
recall or mAP. Review image overlays and source timestamps separately. Keep model-only
and model-plus-inventory area scenarios distinct. Unknown local shade, roof loading,
access rules and current site conditions remain unknown.

## Required evidence

- Revalidate model files, SHA256 and actual inference; no COCO substitution.
- Audit real Geneva feature validity, fields, dates, EGIDs and query coordinates.
- Save images, source URLs, requested/effective bounds and resolution.
- Save full Sonnendach geometries/attributes and one result row per unique roof ID.
- Export gross, occupied, setback and usable areas in horizontal and roof-plane m²,
  plus usable polygons and explicit status/limitations.
- Prove complete image coverage, nonnegative bounded areas, no overlap double-counting,
  and no missing target roof IDs with unit and live integration checks.
- Inspect overlays from every test site; document observed failures rather than
  treating a successful software run as demonstrated physical accuracy.

The model promotion gates remain unchanged and unmet. This work establishes the
requested workflow and its Geneva behaviour, not a production installation approval.
