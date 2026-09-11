from backend.services.planning_service import planning_constraints
"""Reuse detections and packing on independent, physical roof surfaces."""
import hashlib

# Below this a roof is flat enough to need racks rather than flush mounting.
FLAT_PITCH_DEG = 5.0
import math
import time
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.ops import transform, unary_union
from backend.services.runtime_cache import captures
from backend.services.geometry_service import build_usable, polygon_from_points
from backend.services.panel_optimizer import flat_roof_layout, optimise_panels
from backend.services.module_geometry import module_geometry
from backend.services.energy_service import capacity
from backend.services.confidence_service import summarise_confidence
from backend.services.sunlight_service import sunlight_exclusions, public_sunlight
from backend.services.suitability_service import assess_face, dimensions, grouped_panels, building_assessment, sonnendach_comparison, data_provenance
from backend.services import vintage_service

TO_WGS84 = Transformer.from_crs(2056, 4326, always_xy=True)


def parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [p for g in getattr(geometry, "geoms", []) for p in parts(g)]


class GeoReference:
    """Map/image conversion, carrying the capture's own alignment offset.

    The photograph is corrected for terrain, not for building height, so a roof
    appears a metre or two from its coordinates. Both directions apply the same
    measured shift, which keeps geometry read from the map and detections found
    in the image describing the same piece of roof.
    """

    def __init__(self, grid):
        self.x, _, _, self.y = grid["bbox"]
        self.ppm = grid["pixels_per_metre"]
        self.shift_x, self.shift_y = grid.get("shift_px", (0.0, 0.0))

    def world(self, x, y, z=None):
        return (self.x + (np.asarray(x) - self.shift_x)/self.ppm,
                self.y - (np.asarray(y) - self.shift_y)/self.ppm)

    def pixel(self, x, y, z=None):
        return ((np.asarray(x)-self.x)*self.ppm + self.shift_x,
                (self.y-np.asarray(y))*self.ppm + self.shift_y)


def solar_data(props, override=None, performance_ratio=.8):
    def number(key):
        try:
            value = float(props.get(key))
            return value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            return None
    irradiation = number("mstrahlung")
    if irradiation is not None and irradiation > 3000:
        irradiation = None
    return {"irradiation_kwh_m2_year": irradiation, "suitability_class": number("klasse"),
            "specific_yield_kwh_kwp": override if override is not None else
                (irradiation * performance_ratio if irradiation is not None else None),
            "yield_source": "User supplied specific yield" if override is not None else
                (f"Sonnendach irradiation x {performance_ratio:.2f} performance ratio (planning estimate)" if irradiation is not None else None),
            "performance_ratio": performance_ratio if irradiation is not None and override is None else None}


def deduplicate(objects):
    """Combine duplicate evidence without growing PV into adjacent obstacles.

    Manual annotations keep their kind when merged with automatic obstacles.
    Touching boundaries alone do not merge separate detections.
    """
    pending = [{**o, "geometry": polygon_from_points(o["polygon"])} for o in objects]
    pending.sort(key=lambda o: (o["kind"] != "existing_pv", o.get("source") != "manual"))
    merged = []
    for item in pending:
        geometry = item["geometry"]
        if geometry.is_empty:
            continue
        # A reflection entirely inside detected PV is duplicate evidence.
        # Partial overlap with a dormer, awning or vent is not: unioning those
        # different classes made the PV mask spread across the whole terrace.
        if item["kind"] == "skylight" and item.get("source") == "image":
            duplicate = next((old for old in merged if old["kind"] == "existing_pv"
                and old["geometry"].intersection(geometry).area / geometry.area >= .95), None)
            if duplicate is not None:
                duplicate["sources"] = sorted(set(duplicate["sources"] + ["image"]))
                duplicate["kinds"] = sorted(set(duplicate["kinds"] + [item["kind"]]))
                continue
        hits = [old for old in merged if old["kind"] == item["kind"]
                and old["geometry"].intersection(geometry).area > 1e-6]
        if hits:
            first = hits[0]
            geometry = unary_union([geometry] + [h["geometry"] for h in hits])
            sources = sorted(set([item.get("source", "unknown")] + [s for h in hits for s in h["sources"]]))
            for hit in hits:
                merged.remove(hit)
            item = {**first, "geometry": geometry, "sources": sources,
                    "kinds": sorted(set([item["kind"]] + [k for h in hits for k in h["kinds"]])),
                    "height_m": max([item.get("height_m") or 0] + [h.get("height_m") or 0 for h in hits]) or None}
        else:
            item = {**item, "sources": [item.get("source", "unknown")], "kinds": [item["kind"]]}
        merged.append(item)
    return merged


def safe_union(geometries):
    """Union geometry that projection may have left slightly self-inconsistent.

    Projecting a face's local outline back through its own plane can produce a
    ring that touches itself, and GEOS then raises a side-location conflict
    mid-union. Repairing each part first keeps a single bad face from failing
    the whole analysis.
    """
    cleaned = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        if not geometry.is_valid:
            # make_valid over buffer(0): buffer(0) resolves a self-touching ring
            # by discarding a lobe, which would quietly lose real roof area.
            repaired = make_valid(geometry)
            pieces = getattr(repaired, "geoms", [repaired])
            polygons = [g for g in pieces if g.geom_type == "Polygon" and not g.is_empty]
            geometry = unary_union(polygons) if polygons else Polygon()
        if geometry.is_empty:
            continue
        cleaned.append(geometry)
    if not cleaned:
        return Polygon()
    try:
        return unary_union(cleaned)
    except GEOSException:
        # A hairline overlap between two repaired faces can still trip GEOS.
        return unary_union([g.buffer(1e-9) for g in cleaned]).buffer(-1e-9)


def analyse_surfaces(image, settings, objects, warnings, model, start=None):
    start = start or time.perf_counter()
    context = captures.get(settings.capture_id)
    if context is None:
        raise ValueError("Map capture expired. Click the building again to reload its roof model.")
    if (image.size != (context["grid"]["width"], context["grid"]["height"])
            or hashlib.sha256(image.tobytes()).hexdigest() != context["image_hash"]):
        raise ValueError("This image does not match the roof model. Reload the map capture.")
    if not math.isclose(settings.pixels_per_metre or 0, context["grid"]["pixels_per_metre"], rel_tol=1e-5):
        raise ValueError("Map scale is fixed by its georeferencing. Restore the map scale or upload the image separately.")
    # The register says whether an array exists; the image says where. Each is
    # blind where the other sees, so disagreement is worth stating plainly.
    register = context.get("pv_register") or {}
    vintage = context.get("vintage") or {}
    geo = GeoReference(context["grid"])
    warnings = list(dict.fromkeys(context["warnings"] + warnings))
    ids = {f["id"] for f in context["faces"]}
    if set(settings.face_overrides) - ids:
        raise ValueError("An edited face does not belong to this building.")
    merged = deduplicate(objects)
    for obj in merged:
        obj["world"] = transform(geo.world, obj["geometry"])
    clip = transform(geo.world, Polygon(settings.building_override)) if settings.building_override else None
    occupied = Polygon()
    faces = []
    # Overlapping official projections are assigned to the upper face, avoiding
    # physically impossible modules under an overhanging/dormer roof.
    ordered = sorted(context["faces"], key=lambda f: (-f["plane"].origin[2], f["id"]))
    for face in ordered:
        plane = face["plane"]
        world = face["geometry"]
        if face["id"] in settings.face_overrides:
            world = transform(geo.world, Polygon(settings.face_overrides[face["id"]]))
        if clip is not None:
            world = world.intersection(clip)
        overlap = world.intersection(occupied).area
        world = world.difference(occupied)
        occupied = safe_union([occupied, world])
        local = plane.local_geometry(world)
        local_objects = []
        for obj in merged:
            for part in parts(obj["world"].intersection(world)):
                if part.area < 1e-6:
                    continue
                local_objects.append({k: v for k, v in obj.items() if k not in {"geometry", "world", "polygon"}} |
                    {"polygon": list(plane.local_geometry(part).exterior.coords)[:-1]})
        # RWA clearances can reach an adjacent face even when the opening itself
        # does not intersect it. Transform the full footprint before buffering.
        packing_objects = []
        for obj in merged:
            # The same projection is used for every nearby obstacle, including
            # its portion on an adjacent face, so no safety margin disappears.
            for part in parts(obj["world"]):
                packing_objects.append({"kind": obj["kind"], "kinds": obj.get("kinds", []),
                    "polygon": list(plane.local_geometry(part).exterior.coords)[:-1]})
        usable, excluded = build_usable(local, packing_objects, 1, settings)
        angle = 0.
        if not local.is_empty:
            coords = list(local.minimum_rotated_rectangle.exterior.coords)
            a, b = max(zip(coords, coords[1:]), key=lambda pair: math.dist(*pair))
            angle = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
        # Existing alignment control acts as an offset from automatic face alignment.
        angle -= settings.angle - context["default_angle"]
        diagnostics = plane.describe()
        # A flat roof carries tilted racks that shade each other; a pitched one
        # mounts flush and needs no row spacing.
        tilted = diagnostics["pitch_deg"] <= FLAT_PITCH_DEG
        diagnostics["mounting"] = "tilted racks" if tilted else "flush to the pitch"
        physical_panels, orientation = optimise_panels(
            usable, 1, settings.panel, angle, diagnostics, tilted=tilted)
        solar = solar_data(face["properties"], settings.annual_specific_yield, settings.performance_ratio)
        sunlight = face.get("sunlight", {"available": False, "reason": "No surrounding height analysis."})
        assessment = assess_face(plane, solar, sunlight, settings)
        shaded = sunlight_exclusions(sunlight, local, settings.minimum_sun_access)
        panels = physical_panels
        if settings.layout_policy == "recommended":
            if not assessment["eligible"]:
                usable = Polygon()
                panels = []
            elif not shaded.is_empty:
                usable = usable.difference(shaded)
                panels, orientation = optimise_panels(
                    usable, 1, settings.panel, angle, diagnostics, tilted=tilted)
            # Rows separated by their designed rack shadow gap still belong
            # to one array. Previously a two-module row was discarded even
            # when several such rows formed a practical connected installation.
            module_depth = settings.panel.height if orientation == "portrait" else settings.panel.width
            grouping_gap = max(settings.panel.gap, flat_roof_layout(settings.panel, module_depth)[1]) if tilted else settings.panel.gap
            panels = grouped_panels(panels, grouping_gap, settings.minimum_array_panels)
            if not panels and assessment["eligible"] and physical_panels:
                assessment["status"] = "not_recommended"
                assessment["reasons"].append("After shade screening, no sufficiently large connected module group remains.")
        excluded = local.difference(usable)
        if face["id"] in settings.face_overrides and sunlight.get("available"):
            assessment["cautions"].append("Sunlight was sampled on the original face; manually extended areas require a fresh map selection.")
            if assessment["status"] == "suitable":
                assessment["status"] = "needs_review"
        diagnostics.update(overlapping_projection_removed_m2=overlap, alignment_deg=angle)
        faces.append({"id": face["id"], "plane": plane, "world": world, "local": local,
                      "objects": local_objects, "usable": usable, "excluded": excluded,
                      "shaded": shaded, "sunlight": public_sunlight(sunlight), "assessment": assessment,
                      "dimensions": dimensions(local), "physical_panel_count": len(physical_panels),
                      "panels": panels, "orientation": orientation, "diagnostics": diagnostics, "solar": solar,
                      "official_pitch_deg": face["properties"].get("neigung"),
                      "official_azimuth_deg": ((float(face["properties"]["ausrichtung"]) + 180) % 360)
                          if face["properties"].get("ausrichtung") is not None else None})
    available = (all(f["solar"]["specific_yield_kwh_kwp"] is not None for f in faces if f["panels"])
                 and any(f["solar"]["specific_yield_kwh_kwp"] is not None for f in faces))
    if settings.objective == "energy" and not available:
        raise ValueError("Annual-energy optimisation is unavailable: some faces have no irradiation or supplied yield.")
    original_counts = {f["id"]: len(f["panels"]) for f in faces}
    target = (max(0., settings.annual_consumption_kwh-settings.existing_generation_kwh)
              if settings.annual_consumption_kwh is not None and settings.existing_generation_kwh is not None else None)
    def allocation(objective):
        ordered_faces = sorted(faces, key=lambda f: (
            -(f["solar"]["specific_yield_kwh_kwp"] or 0) if objective == "energy" or target is not None else -len(f["panels"]), f["id"]))
        remaining = settings.max_panels if settings.max_panels is not None else sum(original_counts.values())
        energy_remaining = target if available or target == 0 else None
        counts = {}
        for f in ordered_faces:
            wanted = min(remaining, original_counts[f["id"]])
            module_energy = settings.panel.power/1000 * (f["solar"]["specific_yield_kwh_kwp"] or 0)
            if energy_remaining is not None:
                needed = math.ceil(max(0., energy_remaining)/module_energy) if module_energy else 0
                if needed and settings.layout_policy == "recommended":
                    needed = max(needed, settings.minimum_array_panels)
                wanted = min(wanted, needed)
            if settings.layout_policy == "recommended":
                wanted = len(grouped_panels(f["panels"][:wanted], settings.panel.gap, settings.minimum_array_panels))
            counts[f["id"]] = wanted
            remaining -= counts[f["id"]]
            if energy_remaining is not None:
                energy_remaining -= wanted*module_energy
        return counts
    comparisons = {}
    for objective in ["capacity", "energy"] if available else ["capacity"]:
        counts = allocation(objective)
        comparisons[objective] = {"panel_count": sum(counts.values()),
            "additional_kwp": sum(counts.values()) * settings.panel.power / 1000,
            "annual_energy_kwh": round(sum(counts[f["id"]] * settings.panel.power / 1000 *
                (f["solar"]["specific_yield_kwh_kwp"] or 0) for f in faces)) if available else None}
    chosen = allocation(settings.objective)
    features, public_faces, image_panels = [], [], []
    def feature(geometry, layer, face_id, **properties):
        if not geometry.is_empty:
            features.append({"type": "Feature", "properties": {"layer": layer, "face_id": face_id, **properties},
                             "geometry": mapping(transform(TO_WGS84.transform, geometry))})
    for f in faces:
        plane = f["plane"]
        f["panels"] = f["panels"][:chosen[f["id"]]]
        surface_area, projected_area = f["local"].area, f["world"].area
        stats = {"projected_area_m2": round(projected_area, 2), "surface_area_m2": round(surface_area, 2),
                 "usable_area_m2": round(f["usable"].area, 2), "additional_panel_count": len(f["panels"]),
                 "existing_pv_regions": sum(o["kind"] == "existing_pv" for o in f["objects"]),
                 "obstacle_count": sum(o["kind"] != "existing_pv" for o in f["objects"]),
                 **capacity(len(f["panels"]), settings.panel.power, f["solar"]["specific_yield_kwh_kwp"])}
        feature(f["world"], "roof", f["id"], **f["solar"])
        feature(plane.world_geometry(f["usable"]), "usable", f["id"])
        feature(plane.world_geometry(f["excluded"]), "excluded", f["id"])
        feature(plane.world_geometry(f["shaded"]), "shade", f["id"])
        for obj in f["objects"]:
            feature(plane.world_geometry(Polygon(obj["polygon"])), "pv" if obj["kind"] == "existing_pv" else "obstacles", f["id"])
        for ring in f["panels"]:
            world_panel = plane.world_geometry(Polygon(ring))
            feature(world_panel, "panels", f["id"])
            image_panels.append(list(transform(geo.pixel, world_panel).exterior.coords)[:-1])
        public_faces.append({"id": f["id"], **stats, "geometry_source": plane.source,
            "dimensions": f["dimensions"], "sunlight": f["sunlight"], "assessment": f["assessment"],
            "physical_panel_count": f["physical_panel_count"],
            "pitch_deg": f["diagnostics"]["pitch_deg"] if plane.source != "projected_2d" else None,
            "azimuth_deg": f["diagnostics"]["azimuth_deg"] if plane.source != "projected_2d" else None,
            "official_pitch_deg": f["official_pitch_deg"], "official_azimuth_deg": f["official_azimuth_deg"],
            "solar": f["solar"], "orientation": f["orientation"],
            "local_roof": mapping(f["local"]), "local_usable": mapping(f["usable"]),
            "local_objects": f["objects"], "local_panels": f["panels"],
            "panels_3d": [module_geometry(plane, panel, settings.panel,
                            f["diagnostics"]["alignment_deg"],
                            f["diagnostics"]["pitch_deg"] <= FLAT_PITCH_DEG)
                          for panel in f["panels"]],
            "geometry_calculation": "Orthonormal 3D roof plane; physical surface metres; projected to the 2D display",
            "diagnostics": {**f["diagnostics"], **stats}})
    fallback = sum(f["plane"].source == "projected_2d" for f in faces)
    if fallback:
        warnings.append(f"3D roof geometry unavailable or unreliable for {fallback} face(s). Those faces use projected 2D geometry.")
    if settings.boundary_edited or settings.face_overrides:
        warnings.append("Boundary corrections reuse the original fitted planes. Recheck pitch when moving a boundary onto a different surface.")
    if clip is not None:
        warnings.append("The whole-building edit crops official faces; extensions outside them require a separately drawn map roof.")
    warnings.append("Planning estimate: nearby shade is sampled from a static DSM, not an hourly weather simulation. Structure, snow/wind loads and installation compliance remain unassessed.")
    if target is not None and target > 0 and not available:
        warnings.append("The annual demand target cannot be applied without production estimates for the proposed faces.")
    if available:
        warnings.append(f"Annual energy uses face-average irradiation with an assumed {settings.performance_ratio:.0%} performance ratio, or your supplied yield; it is not a production guarantee.")
    surface = sum(f["local"].area for f in faces)
    count = len(image_panels)
    assessment = building_assessment(settings, faces, count, comparisons[settings.objective]["annual_energy_kwh"],
                                     sum(f["physical_panel_count"] for f in faces))
    display_objects = []
    outline = safe_union([f["world"] for f in faces])
    for obj in merged:
        for part in parts(obj["world"].intersection(outline)):
            display_objects.append({k: v for k, v in obj.items() if k not in {"geometry", "world", "polygon"}} |
                {"polygon": list(transform(geo.pixel, part).exterior.coords)[:-1]})
    # Compared on this roof only: the model also sees panels on the buildings
    # next door, and those must not stand in for this building's array.
    seen_pv = sum(1 for o in display_objects if o["kind"] == "existing_pv")
    if register.get("known") and not seen_pv:
        power = register.get("total_power_kw")
        warnings.append(
            "The federal register lists PV on this building"
            + (f" ({power:g} kW)" if power else "")
            + ", but none was found on it in the image. Mark the existing array, "
            "or modules may be proposed where panels already stand."
        )
    elif seen_pv and register.get("egids") and not register.get("known"):
        warnings.append(
            "Panels were found on this roof with no matching entry in the federal "
            "register. Small private arrays are often unregistered, so this is "
            "expected rather than a contradiction."
        )
    warnings = list(dict.fromkeys(warnings))
    return {"roof": mapping(transform(geo.pixel, outline)),
            "usable_area": mapping(transform(geo.pixel, safe_union([f["plane"].world_geometry(f["usable"]) for f in faces]))),
            "excluded_area": mapping(transform(geo.pixel, safe_union([f["plane"].world_geometry(f["excluded"]) for f in faces]))),
            "proposed_panels": image_panels,
            "pv_register": register,
            "planning_constraints": planning_constraints(settings),
            "data_provenance": data_provenance(register, faces, display_objects, model),
            "vintage": vintage,
            "input_confidence": vintage_service.confidence(
                vintage, register,
                sum(1 for f in faces if f["plane"].source != "projected_2d"), len(faces)),
            "existing_pv": [o for o in display_objects if o["kind"] == "existing_pv"],
            "obstacles": [o for o in display_objects if o["kind"] != "existing_pv"],
            "faces": public_faces, "map_overlay": {"type": "FeatureCollection", "features": features},
            "assessment": assessment,
            "shaded_area": mapping(transform(geo.pixel, safe_union([f["plane"].world_geometry(f["shaded"]) for f in faces]))),
            "objective": settings.objective, "energy_available": available,
            "sonnendach": sonnendach_comparison(
                context["faces"], faces, settings,
                comparisons[settings.objective]["annual_energy_kwh"], count),
            "objective_comparison": comparisons,
            "objective_note": "Suitability screening precedes packing. Energy prioritises higher-yield faces under a panel limit. An entered annual demand target also limits the layout; without a limit both objectives use the same eligible space.",
            "model": model, "warnings": list(dict.fromkeys(warnings)), "confidence": summarise_confidence(display_objects), "mode": settings.mode,
            "statistics": {"existing_pv_regions": sum(o["kind"] == "existing_pv" for o in display_objects),
                "additional_panel_count": count, **capacity(count, settings.panel.power),
                "annual_energy_kwh": comparisons[settings.objective]["annual_energy_kwh"],
                "roof_area_m2": round(surface, 2), "surface_area_m2": round(surface, 2),
                "projected_area_m2": round(sum(f["world"].area for f in faces), 2),
                "usable_area_m2": round(sum(f["usable"].area for f in faces), 2),
                "roof_utilisation": round(count*settings.panel.width*settings.panel.height/surface*100, 1) if surface else 0,
                "pixels_per_metre": geo.ppm, "approximate": bool(fallback), "fallback_faces": fallback,
                "orientation": "per face", "elapsed_ms": round((time.perf_counter()-start)*1000)}}
