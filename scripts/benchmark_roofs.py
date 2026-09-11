"""Fixed-roof validation, cache warming and offline result capture.

Run from the project root: python -m scripts.benchmark_roofs --repeats 3
First-pass timings retain existing disk caches; they are NOT cold-download timings.
"""
import argparse
import asyncio
import base64
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree

from backend.main import analyse
from backend.map_routes import shadow
from backend.schemas.analysis import AnalysisSettings
from backend.schemas.map import MapSelection
from backend.services.map_service import prepare_capture
from backend.services.panel_optimizer import optimise_panels
from backend.services.geometry_service import build_usable
from backend.services.runtime_cache import captures

CASES = [
    ("zurich", 47.37796628110402, 8.541166222603481),
    ("bern", 46.95239282008464, 7.428339431895318),
    ("brugg", 47.481398953809006, 8.206642078830201),
]


def validate(result, settings):
    """Check physical placement independently of optimiser control flow."""
    count, naive = 0, 0
    for face in result["faces"]:
        usable = shape(face["local_usable"])
        panels = [Polygon(p) for p in face["local_panels"]]
        tree = STRtree(panels)
        for i, panel in enumerate(panels):
            assert usable.buffer(1e-7).covers(panel), f"Outside usable face {face['id']}"
            tilt = settings.panel.flat_roof_tilt_deg if face['diagnostics'].get('mounting') == 'tilted racks' else 0
            assert abs(panel.area-settings.panel.width*settings.panel.height*np.cos(np.radians(tilt))) < 1e-6
            for j in tree.query(panel):
                if j > i:
                    assert panel.intersection(panels[j]).area < 1e-8, "Overlapping panels"
        count += len(panels)
        # Same module, edge margin and local face, without occupancy or screening.
        roof = shape(face["local_roof"])
        empty_usable, _ = build_usable(roof, [], 1, settings)
        naive += len(optimise_panels(empty_usable, 1, settings.panel, 0)[0])
    assert count == result["statistics"]["additional_panel_count"]
    return {"placement_valid": True, "checked_panels": count,
            "empty_roof_grid_panel_count": naive,
            "additional_panel_count": count,
            "capacity_change_fraction": (count-naive)/naive if naive else None,
            "comparison_note": "Empty-roof grid ignores occupancy and screening. This measures planning impact, not accuracy."}


def percentiles(values):
    return {"p50_seconds": float(np.percentile(values, 50)),
            "p95_seconds": float(np.percentile(values, 95)), "samples": len(values)} if values else None


async def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, first, warm = [], [], []
    for name, lat, lon in CASES:
        row = {"case": name, "latitude": lat, "longitude": lon}
        try:
            selection = MapSelection(latitude=lat, longitude=lon)
            start = time.perf_counter()
            capture = await prepare_capture(selection)
            image_bytes = base64.b64decode(capture["image_base64"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            settings = AnalysisSettings(capture_id=capture["capture_id"], roof=capture["roof"],
                objects=capture["objects"], pixels_per_metre=capture["pixels_per_metre"],
                scale_verified=True, angle=capture["angle"])
            result = analyse(image, settings)
            row["first_pass_seconds"] = time.perf_counter()-start
            first.append(row["first_pass_seconds"])
            row.update(validate(result, settings))
            row["warm_seconds"] = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                repeat = await prepare_capture(selection)
                repeated = analyse(image, settings)
                elapsed = time.perf_counter()-start
                assert repeat["capture_id"] == capture["capture_id"]
                assert repeated["proposed_panels"] == result["proposed_panels"]
                row["warm_seconds"].append(elapsed)
                warm.append(elapsed)
            row["fallback_faces"] = result["statistics"]["fallback_faces"]
            row["pv_regions"] = result["statistics"]["existing_pv_regions"]
            row["shadow_available_faces"] = sum("horizons" in f.get("sunlight", {})
                for f in captures.get(capture["capture_id"])["faces"])
            (output / f"{name}.jpg").write_bytes(image_bytes)
            (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (output / f"{name}-shadow.json").write_text(json.dumps(shadow(capture["capture_id"], 6, 12)), encoding="utf-8")
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"measured_at": datetime.now(timezone.utc).isoformat(), "cases": rows,
        "first_pass_existing_disk_cache": percentiles(first), "warm": percentiles(warm),
        "limitations": "Three fixed roofs, not a representative accuracy sample. Existing disk caches retained. No manually labelled obstacle ground truth; no precision/recall claims. Model warmup included in first pass."}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any("error" in r for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/benchmark")
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=3)
    asyncio.run(main(parser.parse_args()))
