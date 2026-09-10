import math
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from app.geometry import MODULE, esri_polygon, pack_panels, pack_building, roof_transform

@pytest.mark.parametrize('slope,azimuth',[(0,0),(30,0),(30,90),(45,180),(60,270)])
def test_real_module_dimensions_and_roof_clearance(slope,azimuth):
    roof = box(2640000,1230000,2640012,1230010)
    panels,stats = pack_panels(roof,[],slope,azimuth,setback=.3)
    assert panels
    forward,backward = roof_transform(roof,slope,azimuth)
    safe = affinity.affine_transform(roof,forward).buffer(-.3)
    for panel in panels:
        in_plane = affinity.affine_transform(panel,forward)
        assert safe.buffer(1e-7).covers(in_plane)
        assert in_plane.area == pytest.approx(MODULE.length_m*MODULE.width_m,abs=1e-7)
        assert panel.area == pytest.approx(MODULE.length_m*MODULE.width_m*math.cos(math.radians(slope)),abs=1e-7)
        coords = list(in_plane.exterior.coords)
        lengths = sorted(math.dist(coords[i],coords[i+1]) for i in range(4))
        assert lengths == pytest.approx([MODULE.width_m]*2+[MODULE.length_m]*2,abs=1e-7)
    assert stats['surface_area_m2'] == pytest.approx(120/math.cos(math.radians(slope)),abs=.01)
    assert sum(p.area for p in panels) == pytest.approx(unary_union(panels).area)

def test_obstacles_holes_and_concave_roof_never_receive_panels():
    roof = Polygon([(0,0),(12,0),(12,6),(7,6),(7,12),(0,12)],holes=[[(1,1),(1,3),(3,3),(3,1)]])
    obstacles = [box(4,2,6,6),box(5,3,6.5,7)]
    panels,stats = pack_panels(roof,obstacles,25,140,setback=.3)
    forward,_ = roof_transform(roof,25,140)
    blocked = unary_union([affinity.affine_transform(p,forward).buffer(.25) for p in obstacles])
    assert panels
    for panel in panels:
        assert roof.covers(panel)
        assert not affinity.affine_transform(panel,forward).intersects(blocked)

def test_fully_blocked_or_tiny_roof_has_no_panels():
    assert pack_panels(box(0,0,8,8),[box(-1,-1,9,9)],30,180)[0] == []
    assert pack_panels(box(0,0,.5,.5),[],0,0)[0] == []

def test_duplicate_exclusions_do_not_reduce_area_twice():
    roof,obstacle = box(0,0,10,10),box(3,3,5,5)
    a,sa = pack_panels(roof,[obstacle],20,180)
    b,sb = pack_panels(roof,[obstacle,obstacle],20,180)
    assert len(a)==len(b) and sa==sb

def test_esri_multiple_exteriors_and_holes():
    geom = esri_polygon({'rings':[list(box(0,0,10,10).exterior.coords),list(box(1,1,2,2).exterior.coords),list(box(20,20,21,21).exterior.coords)]})
    assert geom.area == 100
    assert not geom.covers(box(1,1,2,2))

def test_invalid_pitch_rejected():
    with pytest.raises(ValueError):
        pack_panels(box(0,0,10,10),[],90,0)

def test_overlapping_facets_and_duplicate_faces_cannot_stack_panels():
    faces=[dict(geometry=box(0,0,12,12),slope=25,azimuth=180),
           dict(geometry=box(5,0,16,12),slope=35,azimuth=90),
           dict(geometry=box(0,0,12,12),slope=25,azimuth=180)]
    layouts=pack_building(faces,[])
    assert layouts[0][0] and layouts[1][0] and not layouts[2][0]
    panels=[p for group,_ in layouts for p in group]
    assert sum(p.area for p in panels)==pytest.approx(unary_union(panels).area,abs=1e-7)
    for face,(group,_) in zip(faces,layouts):
        assert all(face['geometry'].buffer(1e-7).covers(p) for p in group)

def test_realistic_row_gaps_and_flat_roof_density():
    roof=box(0,0,10,10)
    panels,_=pack_panels(roof,[],0,0)
    assert panels
    for i,panel in enumerate(panels):
        assert all(panel.distance(p)>.099999 for p in panels[i+1:])
    # Full flat modules plus 1 m row spacing cannot tile most of the roof.
    assert sum(p.area for p in panels)<roof.area*.65

def test_empty_detected_roof_cannot_create_panels():
    assert pack_building([dict(geometry=Polygon(),slope=30,azimuth=180)],[])[0][0]==[]
