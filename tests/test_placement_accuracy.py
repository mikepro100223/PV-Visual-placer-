import math
import numpy as np
import pytest
from shapely.geometry import box, Polygon
from backend.schemas.analysis import AnalysisSettings, PanelConfig
from backend.services.geometry_service import build_usable
from backend.services.module_geometry import module_geometry
from backend.services.roof_plane import RoofPlane
from backend.services.panel_optimizer import optimise_panels, flat_roof_layout


def test_chimney_on_neighbouring_face_keeps_its_clearance():
    roof = box(0,0,10,10)
    settings = AnalysisSettings(roof=list(roof.exterior.coords),edge_margin=0,obstacle_margin=.4)
    obstacle = {"kind":"chimney","polygon":list(box(10.1,4,11,6).exterior.coords)}
    usable,_ = build_usable(roof,[obstacle],1,settings)
    assert not usable.covers(box(9.8,4.5,10,5.5))
    assert usable.area == pytest.approx(100-.3*2.8)


@pytest.mark.parametrize('angle',[0,17,90])
def test_tilted_rack_has_real_module_dimensions_and_normal(angle):
    from shapely.affinity import rotate
    panel=PanelConfig()
    depth,_=flat_roof_layout(panel,panel.height)
    ring=list(rotate(box(0,0,panel.width,depth),angle,origin=(0,0)).exterior.coords)[:-1]
    plane=RoofPlane.from_slopes(2600000,1200000,500,source='swisssurface3d')
    result=module_geometry(plane,ring,panel,angle,True)
    corners=np.array(result['corners_lv95_ln02'])
    lengths=sorted(np.linalg.norm(corners[(i+1)%4]-corners[i]) for i in range(4))
    assert lengths == pytest.approx(sorted([panel.width,panel.width,panel.height,panel.height]))
    assert np.ptp(corners[:,2]) == pytest.approx(panel.height*math.sin(math.radians(15)))
    normal=np.array(result['normal'])
    assert np.linalg.norm(normal) == pytest.approx(1)
    assert np.dot(normal,corners[1]-corners[0]) == pytest.approx(0,abs=1e-8)


def test_boundary_phases_fit_a_narrow_usable_strip():
    # A narrow usable lane starts at a phase not present in the old 4x4 grid.
    from shapely.ops import unary_union
    usable=unary_union([box(0,0,.1,5),box(1.13,0,2.13,6)])
    panel=PanelConfig(width=1,height=2,gap=0)
    rings,_=optimise_panels(usable,1,panel)
    assert len(rings)==3
    assert all(usable.buffer(1e-8).covers(Polygon(r)) for r in rings)


def test_short_rack_rows_form_one_practical_array():
    from backend.services.suitability_service import grouped_panels
    panel=PanelConfig()
    depth,gap=flat_roof_layout(panel,panel.height)
    rings=[list(box(x*(panel.width+panel.gap),y*(depth+gap),
                    x*(panel.width+panel.gap)+panel.width,y*(depth+gap)+depth).exterior.coords)[:-1]
           for x in range(2) for y in range(3)]
    assert not grouped_panels(rings,panel.gap,4)
    assert len(grouped_panels(rings,gap,4))==6
