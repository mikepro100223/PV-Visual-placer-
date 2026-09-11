"""Lift rack footprints into actual 3D module surfaces."""
import math
import numpy as np


def module_geometry(plane, ring, panel, alignment_deg, tilted):
    angle = math.radians(alignment_deg)
    tilt = math.radians(panel.flat_roof_tilt_deg if tilted else 0.)
    depth = -math.sin(angle)*plane.u + math.cos(angle)*plane.v
    normal = plane.normal*math.cos(tilt) - depth*math.sin(tilt)
    along_depth = [-math.sin(angle)*x + math.cos(angle)*y for x,y in ring]
    low = min(along_depth)
    corners = [plane.xyz(*point) + plane.normal*(t-low)*math.tan(tilt)
               for point,t in zip(ring,along_depth)]
    return {"corners_lv95_ln02": [p.tolist() for p in corners]
                if plane.describe()["height_is_absolute"] else None,
            "normal": normal.tolist(), "surface_corners_m": ring,
            "mounting_tilt_deg": math.degrees(tilt),
            "mounting": "tilted racks" if tilted else "flush to the pitch"}
