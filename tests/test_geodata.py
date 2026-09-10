from shapely.geometry import box
from app.geodata import connected_roof_parts


def record(id,rect,egid=123):
    return dict(featureId=id,attributes=dict(gwr_egid=egid,building_id=id),
                geometry=dict(rings=[list(rect.exterior.coords)]))


def test_connected_parts_include_same_house_without_neighbour_or_separate_outbuilding():
    selected=record(1,box(0,0,10,10))
    wing=record(2,box(10,0,20,10))
    far_wing=record(3,box(20,0,30,10))
    neighbour=record(4,box(0,10,10,20),456)
    separate=record(5,box(40,0,50,10))
    result=connected_roof_parts([selected],[far_wing,separate,neighbour,selected,wing])
    assert {r['featureId'] for r in result}=={1,2,3}


def test_unknown_building_register_id_does_not_merge_neighbours():
    selected=record(1,box(0,0,10,10),0)
    assert connected_roof_parts([selected],[record(2,box(10,0,20,10),0)])==[selected]
