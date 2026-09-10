# Geneva LV95 mapping audit

Read-only audit of the saved artifacts in `artifacts/geneva/{dufour,hug,meyrin}`.
The audit was run at `2026-09-10T13:58:26Z`; it did not download imagery or run
either model.  It independently re-queried the exact `query_bbox2056` recorded in
each report through the official [GeoAdmin identify API](https://docs.geo.admin.ch/access-data/identify-features.html)
with `returnGeometry=true`, `geometryFormat=geojson`, and `sr=2056`.  The source
geometry comparison used the same layer's [Get Features API](https://docs.geo.admin.ch/access-data/get-features.html).

## Source completeness and spatial checks

| Site | Query extent | Saved roofs | Fresh roofs | ID / geometry / properties match | Saved source geometries intersect query | Zero-area / invalid | Positive-area overlap pairs |
|---|---:|---:|---:|---|---:|---:|---:|
| Dufour | 100 m × 100 m | 39 | 39 | 39 / 39 / 39 | 39 / 39 | 0 / 0 | 0 |
| Hug | 100 m × 100 m | 47 | 47 | 47 / 47 / 47 | 47 / 47 | 0 / 0 | 0 |
| Meyrin | 100 m × 100 m | 111 | 111 | 111 / 111 / 111 | 111 / 111 | 21 / 21 | 0 |

The live ID sets are exact matches.  The saved geometry and property payloads are
also exact matches for the fresh response, so there is no evidence of silent
pagination truncation or artifact-side feature loss.  Pairwise intersections have
zero positive area; some planes touch along boundaries (70 Dufour pairs, 99 Hug
pairs, 227 Meyrin pairs including degenerate geometries).

The 21 Meyrin failures are genuine upstream degenerate geometries, not a local
conversion loss.  For example, official feature `19660623` returns the same
GeoJSON and ESRI geometry from Get Features:

```text
[[[[2498181.2,1119069.4], [2498177.4,1119056.4],
   [2498177.4,1119056.4], [2498181.2,1119069.4],
   [2498181.2,1119069.4]]]]
```

It has official `FLAECHE=1.0542933239 m²`, but its returned polygon has zero
planimetric area and Shapely reports “Too few points in geometry component”.
Requests with `geometryPrecision=3` and `precision=3` are ignored by the service;
the documented Get Features parameters expose no higher-precision geometry option.
The coordinates are one-decimal LV95 values, so these narrow roof facets are
already collapsed in the upstream response.  They must remain explicit errors,
not fabricated or silently dropped roofs.

## Image-bbox and north-up mapping

Each `overview.jpg` has the report's exact dimensions.  The union of the saved
SWISSIMAGE tile bboxes equals the report `image_bbox2056` exactly in all three
sites; every tile records `gsd_m=[0.1,0.1]` and `north_up=true`.

| Site | Report image size | Tiles | Image bbox span (m) | Tile union equals bbox | Overview dimensions |
|---|---:|---:|---:|---|---|
| Dufour | 1717 × 1672 | 4 | 171.7 × 167.2 | yes | 1717 × 1672 |
| Hug | 1858 × 1919 | 9 | 185.8 × 191.9 | yes | 1858 × 1919 |
| Meyrin | 1306 × 1326 | 4 | 130.6 × 132.6 | yes | 1306 × 1326 |

Thus the recorded pixel contract is 0.1 m per pixel, with row zero on the
maximum-northing edge.  Visual review of all three overview images shows the
expected LV95 roof/obstacle overlays on the SWISSIMAGE chips; it is a wiring and
registration check only, not a model-accuracy claim.

## Downstream usable-roof caveats

The source inventory is complete, but `usable-roofs-lv95.geojson` is a downstream
product and is not equivalent to the source footprint inventory.  Non-null saved
usable geometries intersect the query for Hug (47/47) and Meyrin (27/27).  Dufour
has 38 non-null geometries; IDs `19616062` and `19623494` do not intersect the
query because the retained usable portions lie just outside its edge, although
their official source roofs do intersect the query by approximately 0.370 and
0.561 m² respectively. This is intentional for this workflow: the query selects
whole roof IDs, then additional imagery covers each complete roof. The reported
usable geometry is not clipped back to the selection box. A separate analysis
of only the area inside that box would need explicit clipping and different
area semantics; it must not silently replace the whole-roof result here.

The usable file also contains null geometry for one Dufour feature and 84 Meyrin
features.  Null entries cannot be independently proven to intersect a bbox; they
are retained here as explicit per-roof outcomes rather than treated as missing
source IDs.  The report's model gates and Geneva inventory comparison remain
experimental and are intentionally not interpreted as accuracy or recall.

## Official area versus returned geometry

`FLAECHE` is the official physical/sloped roof-plane area; a planar EPSG:2056
polygon area is a horizontal footprint.  Comparing `FLAECHE` with
`planimetric_area / cos(NEIGUNG)` gives the relevant dimensional check:

| Site | Valid source geometries | Median relative difference | Mean absolute difference | Range |
|---|---:|---:|---:|---:|
| Dufour | 39 | +0.07% | 0.88% | −4.01% … +3.33% |
| Hug | 47 | +0.05% | 0.44% | −1.60% … +1.41% |
| Meyrin | 90 | −0.86% | 26.69% | −80.04% … +281.78% |

Meyrin's large spread is concentrated in very small, quantized facets: restricting
to valid footprints at least 5 m² reduces its mean absolute difference to 2.81%
(46 roofs), and at least 10 m² to 1.49% (19 roofs).  The area discrepancy is
therefore a source-geometry/rounding caveat for tiny planes, not evidence that
`FLAECHE` is a planar area.  No area is substituted or silently repaired in this
audit.
