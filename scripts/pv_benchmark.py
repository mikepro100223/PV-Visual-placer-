"""Test existing-array detection over many real roofs known to carry one.

The federal plant register says which buildings have PV and how much, so a
large sample needs no hand labelling. Two things are measured per roof:

  recall   - how much array was found against what the recorded capacity implies
  conflict - how many of the modules the app proposes sit on pixels that look
             like the array it found on that same roof

Conflict is the error a user actually sees: new modules drawn on top of modules
already installed. It needs no ground truth, because the roof supplies its own
reference - the array the detector did find.
"""

import argparse
import asyncio
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services import map_service as ms

BASE = "http://127.0.0.1:8000"
MAP = "https://api3.geo.admin.ch/rest/services/all/MapServer"
REGISTER = "ch.bfe.elektrizitaetsproduktionsanlagen"
TO_SWISS = Transformer.from_crs(4326, 2056, always_xy=True)
PANEL = {"width": 1.762, "height": 1.134, "power": 450, "gap": 0.02}
M2_PER_KW = 1.0 / 0.19

# Towns to draw the sample from, so the result is not one city's roof styles.
PLACES = [
    ("Zurich", 47.3769, 8.5417), ("Winterthur", 47.5000, 8.7241),
    ("Bern", 46.9480, 7.4474), ("Basel", 47.5596, 7.5886),
    ("Luzern", 47.0502, 8.3093), ("StGallen", 47.4245, 9.3767),
    ("Aarau", 47.3910, 8.0455), ("Lausanne", 46.5197, 6.6323),
    ("Sion", 46.2331, 7.3606), ("Chur", 46.8508, 9.5320),
]
# A proposed module counts as a conflict when its pixels look like the array
# found on the same roof: as blue, and no brighter.
BLUE_TOLERANCE = 4.0
GREY_TOLERANCE = 12.0


async def plants(client, lat, lon, low, high, half=2000, limit=80):
    x, y = TO_SWISS.transform(lon, lat)
    r = await client.get(MAP + "/identify", params={
        "geometry": f"{x-half},{y-half},{x+half},{y+half}",
        "geometryType": "esriGeometryEnvelope", "layers": "all:" + REGISTER,
        "sr": 2056, "geometryFormat": "geojson", "returnGeometry": "true",
        "tolerance": 0, "mapExtent": f"{x-half*2},{y-half*2},{x+half*2},{y+half*2}",
        "imageDisplay": "1000,1000,96", "lang": "en", "limit": limit})
    r.raise_for_status()
    out = []
    for f in r.json().get("results", []):
        p = f["properties"]
        if p.get("sub_category_en") != "Photovoltaic" or not f.get("geometry"):
            continue
        try:
            kw = float(str(p.get("total_power") or "").split()[0].replace(",", "."))
        except (ValueError, IndexError):
            continue
        if low <= kw <= high:
            out.append((kw, shape(f["geometry"]).representative_point(), p))
    return out


def signature(image, polygons):
    """Mean blue-minus-red and brightness of the pixels these polygons cover."""
    mask = np.zeros(image.shape[:2], np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
    picked = mask.astype(bool)
    if picked.sum() < 20:
        return None
    rgb = image.astype(np.float32)
    return (float(np.median((rgb[..., 2] - rgb[..., 0])[picked])),
            float(np.median(rgb.mean(2)[picked])))


def conflicts(image, proposed, arrays):
    """How many proposed modules sit on pixels that look like a found array."""
    reference = signature(image, arrays)
    if reference is None or not proposed:
        return 0, len(proposed)
    blue_ref, grey_ref = reference
    hits = 0
    for quad in proposed:
        here = signature(image, [quad])
        if here is None:
            continue
        if (here[0] >= blue_ref - BLUE_TOLERANCE
                and here[1] <= grey_ref + GREY_TOLERANCE):
            hits += 1
    return hits, len(proposed)


async def one(client, kw, point, props, render):
    lon, lat = ms.TO_WGS84.transform(point.x, point.y)
    cap = await client.post(BASE + "/api/map/prepare",
                            json={"latitude": lat, "longitude": lon})
    if cap.status_code != 200:
        return {"error": f"prepare {cap.status_code}"}
    cap = cap.json()
    if len(cap.get("roof") or []) < 3:
        return None
    image_bytes = base64.b64decode(cap["image_base64"])
    settings = {"roof": cap["roof"], "objects": cap["objects"], "mode": "recommended",
                "panel": PANEL, "pixels_per_metre": cap["pixels_per_metre"],
                "approximate_roof_width": 12, "scale_verified": True,
                "angle": cap["angle"], "use_ai": True, "edge_margin": .3,
                "obstacle_margin": .4, "pv_margin": .2,
                "capture_id": cap["capture_id"]}
    res = await client.post(BASE + "/api/analyse",
                            files={"image": ("r.jpg", image_bytes, "image/jpeg")},
                            data={"settings": json.dumps(settings)})
    if res.status_code != 200:
        return {"error": f"analyse {res.status_code} {res.text[:100]}"}
    d = res.json()
    ppm = cap["pixels_per_metre"]
    image = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
    arrays = [o["polygon"] for o in d["existing_pv"]]
    found = sum(max(Polygon(p).buffer(0).area, 0) for p in arrays) / ppm ** 2
    clash, total = conflicts(image, d["proposed_panels"], arrays)
    if render:
        vis = image.copy()
        for q in d["proposed_panels"]:
            cv2.polylines(vis, [np.array(q, np.int32)], True, (90, 255, 90), 1)
        for o in d["obstacles"]:
            cv2.polylines(vis, [np.array(o["polygon"], np.int32)], True, (255, 150, 0), 1)
        for p in arrays:
            cv2.polylines(vis, [np.array(p, np.int32)], True, (0, 130, 255), 2)
        cv2.polylines(vis, [np.array(cap["roof"], np.int32)], True, (255, 255, 255), 1)
        Image.fromarray(vis).save(render)
    expected = kw * M2_PER_KW
    return {"address": (props.get("address") or "")[:30], "kw": kw,
            "expected_m2": round(expected, 1), "found_m2": round(found, 1),
            "recall": round(found / expected, 2) if expected else None,
            "proposed": total, "conflict": clash,
            "conflict_pct": round(100 * clash / total, 1) if total else 0.0,
            "lat": round(lat, 6), "lon": round(lon, 6)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--per-place", type=int, default=4)
    ap.add_argument("--low", type=float, default=15)
    ap.add_argument("--high", type=float, default=400)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    async with httpx.AsyncClient(timeout=900) as client:
        for place, lat, lon in PLACES:
            try:
                near = await plants(client, lat, lon, args.low, args.high)
            except httpx.HTTPError as exc:
                print(f"{place}: register unavailable ({exc})")
                continue
            near.sort(key=lambda p: -p[0])
            seen = []
            for kw, point, props in near:
                if len(seen) >= args.per_place:
                    break
                if any(point.distance(q) < 40 for q in seen):
                    continue
                seen.append(point)
                name = f"{place}-{len(seen)}"
                try:
                    row = await one(client, kw, point, props, str(out / f"{name}.png"))
                except (httpx.HTTPError, ValueError) as exc:
                    row = {"error": f"{type(exc).__name__} {exc}"}
                if not row:
                    continue
                row["name"] = name
                rows.append(row)
                if "error" in row:
                    print(f"{name:<14} ERROR {row['error']}")
                else:
                    print(f"{name:<14}{row['kw']:>7.0f}kW{row['found_m2']:>9.0f}"
                          f"{(row['recall'] or 0):>7.2f}{row['proposed']:>7}"
                          f"{row['conflict']:>7}{row['conflict_pct']:>7.1f}%")
    (out / "benchmark.json").write_text(json.dumps(rows, indent=1))
    good = [r for r in rows if "error" not in r]
    if not good:
        return
    proposed = sum(r["proposed"] for r in good)
    clash = sum(r["conflict"] for r in good)
    scored = [r["recall"] for r in good if r["recall"] is not None]
    missed = [r for r in good if r["recall"] is not None and r["recall"] < 0.5]
    print("-" * 62)
    print(f"roofs {len(good)} | median recall {np.median(scored):.2f}"
          f" | found <50% on {len(missed)}")
    print(f"proposed {proposed} modules | {clash} on what looks like existing "
          f"array ({100*clash/proposed if proposed else 0:.1f}%)")
    for r in sorted(good, key=lambda q: -q["conflict_pct"])[:6]:
        if r["conflict_pct"] > 0:
            print(f"   {r['name']:<14}{r['conflict_pct']:>6.1f}%  "
                  f"{r['conflict']}/{r['proposed']}  {r['address']}")


asyncio.run(main())
