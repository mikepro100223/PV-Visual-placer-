from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from backend.services.pv_segmentation import corroborated_pv, merge_detections


def obj(g, kind='existing_pv'):
    return dict(polygon=list(g.exterior.coords),kind=kind,confidence=.8,source='yolo')


def test_second_orientation_requires_original_support():
    original=[obj(box(0,0,10,10))]
    supported=obj(box(0,0,20,10))
    assert corroborated_pv(original,[supported,obj(box(30,0,40,10)),obj(box(0,0,100,100)),obj(box(0,0,10,10),'skylight')]) == [supported]
    assert not corroborated_pv([], [supported])


def test_merging_array_banks_preserves_courtyard():
    banks=[box(0,0,12,3),box(0,9,12,12),box(0,0,3,12),box(9,0,12,12)]
    # Overlapping masks can make a ring. No returned exterior may fill it.
    result=merge_detections([obj(unary_union([banks[0],banks[1],banks[2]])),
                             obj(unary_union([banks[0],banks[1],banks[3]]))])
    geometry=unary_union([Polygon(o['polygon']) for o in result])
    assert geometry.symmetric_difference(unary_union(banks)).area < 1e-6
    assert not geometry.covers(Point(6,6))


def test_deployed_model_recovers_bachstrasse_rows_without_claiming_shadow():
    from backend.services.yolo_service import YoloService
    image=Image.open(Path(__file__).parent/'fixtures/bach-pv.jpg').convert('RGB')
    objects,warnings,status=YoloService().detect(image)
    assert status['available'], warnings
    geometry=unary_union([Polygon(o['polygon']).buffer(0) for o in objects if o['kind']=='existing_pv'])
    # Visually reviewed module interiors, expressed on the 950px review image.
    scale=image.width/950
    for x,y in [(425,399),(500,389),(580,338)]:
        assert geometry.covers(Point(x*scale,y*scale))
    # Tall-wing shadow, machinery, roof opening and bare roof.
    for x,y in [(440,275),(670,287),(692,490),(760,285)]:
        assert not geometry.covers(Point(x*scale,y*scale))
