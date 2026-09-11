import cv2
import numpy as np
from shapely.geometry import box, Polygon
from backend.services.hvac_service import detect


def fan_bank():
    image=np.full((220,320,3),95,np.uint8)
    image[65:115,60:190]=215
    for x in range(70,181,12):
        for y in [78,100]:
            cv2.circle(image,(x,y),4,(45,45,45),-1)
            cv2.circle(image,(x,y),1,(180,180,180),-1)
    return image


def test_repeated_fans_exclude_housing_not_only_individual_blades():
    found=detect(fan_bank(),box(0,0,320,220),10)
    assert found
    assert sum(o['fan_count'] for o in found) >= 12
    assert all(o['geometry'].area > 1000 for o in found)


def test_known_pv_is_not_reclassified_as_machinery():
    assert not detect(fan_bank(),box(0,0,320,220),10,[box(50,55,200,125)])


def test_courtyard_furniture_is_outside_detection_scope():
    roof=Polygon(box(0,0,320,220).exterior,[box(50,55,200,125).exterior])
    assert not detect(fan_bank(),roof,10)


def test_isolated_spots_do_not_become_a_fan_bank():
    image=np.full((220,320,3),180,np.uint8)
    for x in [50,150,250]:cv2.circle(image,(x,100),4,(40,40,40),-1)
    assert not detect(image,box(0,0,320,220),10)


def test_aarau_swissimage_fan_banks_are_both_recognised():
    from pathlib import Path
    from PIL import Image
    from shapely.geometry import Point
    image=np.asarray(Image.open(Path(__file__).parent/'fixtures/aarau-fan-banks.png').convert('RGB'))
    found=detect(image,box(0,0,image.shape[1],image.shape[0]),10)
    assert len(found)==2
    # Reviewed points on each of the two visible fan banks, in this crop.
    assert any(o['geometry'].covers(Point(60,45)) for o in found)
    assert any(o['geometry'].covers(Point(60,85)) for o in found)
    assert all(o['geometry'].area < image.shape[0]*image.shape[1]/2 for o in found)
