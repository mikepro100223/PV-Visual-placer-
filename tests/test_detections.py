import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box,Polygon
from shapely.ops import unary_union
from app.detections import (
    arbitrate_detections,
    blocks_placement,
    merge_detections,
    normalise,
    pixel_to_world,
    public_detection,
    world_to_pixel,
)
from app.obstacles.views import detection_views


def obj(geometry,kind='chimney',score=.8,source='rid'):
    return dict(geometry=geometry,kind=kind,confidence=score,source=source)


def test_tile_offset_and_north_up_coordinates_round_trip():
    bounds=(2640000,1230000,2640100,1230080)
    world=pixel_to_world(box(10,20,30,40),bounds,(1000,800),(200,100))
    assert world.bounds==pytest.approx((2640021,1230066,2640023,1230068))
    assert world_to_pixel(world,bounds,(1000,800)).bounds==pytest.approx((210,120,230,140))


def test_padding_outside_actual_image_cannot_become_an_obstacle():
    assert pixel_to_world(box(450,0,510,50),(0,0,40,40),(400,400)).is_empty


def test_aliases_holes_and_repaired_parts_survive_serialisation():
    hole=box(0,0,10,10).difference(box(3,3,7,7))
    d=normalise([obj(hole,'unknown_obstacle',None,'abbas_height')])[0]
    public=public_detection(d)
    assert public['kind']=='other_obstacle' and public['confidence'] is None
    assert Polygon(public['polygon'],public['holes']).area==84
    bow=Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)])
    assert sum(o['geometry'].area for o in normalise([obj(bow)]))==2


def test_duplicate_masks_keep_union_without_expanding_pv_into_obstacles():
    results=merge_detections([obj(box(0,0,10,10),'existing_pv',.9),
                             obj(box(4,0,12,10),'pv_installation',.7),
                             obj(box(9,5,14,10),'dormer',.8)])
    pv=[o for o in results if o['kind']=='pv_installation']
    assert len(pv)==1 and pv[0]['geometry'].area==120 and pv[0]['confidence']==.9
    assert len(results)==2


def test_overlapping_tile_masks_preserve_complete_object_coverage():
    results=merge_detections([
        obj(box(0,0,10,10),score=.9),
        obj(box(4,0,14,10),score=.8),
        obj(box(8,0,18,10),score=.7),
    ])
    chimneys=[o for o in results if o['kind']=='chimney']
    assert len(chimneys)==1
    assert chimneys[0]['geometry'].bounds==(0,0,18,10)
    assert chimneys[0]['geometry'].area==180


def test_merged_detection_uses_measured_source_independent_of_input_order():
    model=obj(box(0,0,2,2),score=.9,source='rid')
    measured=obj(box(0,0,2,2),score=None,source='abbas_height')
    forward=merge_detections([model,measured])[0]
    reverse=merge_detections([measured,model])[0]
    assert forward['source']==reverse['source']=='abbas_height'
    assert forward['model']==reverse['model']=='abbas_height'
    assert forward['sources']==reverse['sources']==['abbas_height','rid']


def test_trusted_pv_removes_only_conflicting_model_obstacle_area():
    pv=obj(box(0,0,10,10),'pv_installation',.9,'swiss')
    rid=obj(box(5,0,12,10),'other_obstacle',.8,'rid')
    measured=obj(box(2,2,4,4),'chimney',None,'abbas_height')
    results=arbitrate_detections([pv,rid,measured])
    clipped=next(o for o in results if o['kind']=='other_obstacle')
    kept=next(o for o in results if o['kind']=='chimney')
    assert clipped['geometry'].equals(box(10,0,12,10))
    assert kept['geometry'].equals(measured['geometry'])
    assert unary_union([o['geometry'] for o in results if o['kind']!='pv_installation']).intersection(pv['geometry']).area==4


def test_shadow_is_advisory_and_public_contract_exposes_that():
    shadow=normalise([obj(box(0,0,2,2),'shadow',.8)])[0]
    chimney=normalise([obj(box(0,0,2,2),'chimney',.8)])[0]
    assert not blocks_placement(shadow)
    assert blocks_placement(chimney)
    assert public_detection(shadow)['blocks_placement'] is False


def test_views_cover_image_and_bound_cost():
    image=Image.new('RGB',(1800,1400))
    views=list(detection_views(image))
    assert views[0][0] is image and len(views)<=17
    covered=unary_union([box(x,y,x+v.width,y+v.height) for v,x,y in views[1:]])
    assert covered.covers(box(0,0,1800,1400))
    assert list(detection_views(Image.new('RGB',(400,400))))[0][0].size==(512,512)


@pytest.mark.parametrize('kind,score',[('car',.9),('chimney',float('nan')),('dormer',2)])
def test_invalid_class_or_confidence_rejected(kind,score):
    with pytest.raises(ValueError):normalise([obj(box(0,0,2,2),kind,score)])
