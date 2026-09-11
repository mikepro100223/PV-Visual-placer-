from typing import Literal
from pydantic import BaseModel, Field, model_validator
import math

Point = tuple[float, float]
MAX_OBJECT_VERTICES = 4096


class PanelConfig(BaseModel):
    width: float = Field(1.134, ge=0.5, le=3)
    height: float = Field(1.762, ge=0.5, le=4)
    power: float = Field(450, ge=50, le=1000)
    gap: float = Field(0.02, ge=0, le=0.5)
    # Ballasted racks on a flat roof. A pitched roof mounts flush and ignores
    # both of these: its modules follow the roof and need no row spacing.
    flat_roof_tilt_deg: float = Field(15, ge=0, le=45)
    # Sun altitude the row spacing is designed against. The default is noon at
    # the winter solstice on the Swiss plateau, the usual conservative choice.
    design_sun_altitude_deg: float = Field(19.2, ge=5, le=60)
    row_gap_m: float | None = Field(None, ge=0, le=10)


class MarkedObject(BaseModel):
    source: Literal["manual", "map", "elevation", "image", "terrain"] = "manual"
    # Segmented array boundaries are much more detailed than a hand-drawn roof.
    # Keep their concavities instead of truncating them or filling their hull.
    polygon: list[Point] = Field(min_length=3, max_length=MAX_OBJECT_VERTICES)
    kind: Literal["existing_pv", "chimney", "skylight", "rwa", "other_obstacle"] = (
        "other_obstacle"
    )
    # Metres the superstructure rises above its roof face, when measured.
    height_m: float | None = Field(None, ge=0, le=50)


class AnalysisSettings(BaseModel):
    capture_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")
    # Edited whole-building outlines crop the original faces. A face override
    # changes only that face, retaining its measured plane and all other faces.
    boundary_edited: bool = False
    edited_face_id: str | None = Field(None, max_length=80)
    face_overrides: dict[str, list[Point]] = Field(default_factory=dict, max_length=100)
    building_override: list[Point] | None = Field(None, min_length=3, max_length=200)
    objective: Literal["capacity", "energy"] = "capacity"
    max_panels: int | None = Field(None, ge=1, le=10000)
    layout_policy: Literal["recommended", "physical"] = "recommended"
    minimum_irradiation: float = Field(800, ge=0, le=2000)
    minimum_sun_access: float = Field(.6, ge=0, le=1)
    minimum_array_panels: int = Field(4, ge=1, le=30)
    annual_consumption_kwh: float | None = Field(None, ge=0, le=10_000_000)
    existing_generation_kwh: float | None = Field(None, ge=0, le=10_000_000)
    roof: list[Point] = Field(min_length=3, max_length=200)
    pixels_per_metre: float | None = Field(None, ge=0.5, le=2000)
    approximate_roof_width: float = Field(12, ge=1, le=200)
    scale_verified: bool = False
    mode: Literal["conservative", "recommended", "maximum"] = "recommended"
    panel: PanelConfig = Field(default_factory=PanelConfig)
    objects: list[MarkedObject] = Field(default_factory=list, max_length=500)
    angle: float = Field(0, ge=-180, le=180)
    annual_specific_yield: float | None = Field(None, ge=0, le=3000)
    performance_ratio: float = Field(0.8, ge=0.1, le=1)
    edge_margin: float = Field(0.3, ge=0, le=3)
    obstacle_margin: float = Field(0.4, ge=0, le=3)
    pv_margin: float = Field(0.2, ge=0, le=3)
    use_ai: bool = True

    @model_validator(mode="after")
    def finite_coordinates(self):
        boundaries = [self.roof] + list(self.face_overrides.values()) + ([self.building_override] if self.building_override else [])
        for polygon in boundaries:
            if not 3 <= len(polygon) <= 200:
                raise ValueError("Polygons need 3 to 200 vertices")
        polygons = boundaries + [o.polygon for o in self.objects]
        if sum(map(len, polygons)) > 100_000:
            raise ValueError("The analysis contains too many polygon vertices")
        for polygon in polygons:
            if any(not math.isfinite(v) for point in polygon for v in point):
                raise ValueError("Coordinates must be finite")
        return self
