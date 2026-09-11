import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box
from app.roof_alignment import align_roof_faces


def aerial():
    data=np.full((400,400,3),30,dtype=np.uint8)
    data[100:300,100:300]=(170,100,65)
    return Image.fromarray(data)


def test_recovers_translation_without_scaling_or_rotating_faces():
    roof=box(9,12,29,32)
    aligned,report=align_roof_faces([roof],aerial(),(0,0,40,40))
    assert report['applied']
    assert report['east_m']==pytest.approx(1,abs=.2)
    assert report['north_m']==pytest.approx(-2,abs=.2)
    assert aligned[0].area==pytest.approx(roof.area)
    assert aligned[0].centroid.distance(box(10,10,30,30).centroid)<.2


def test_shrinks_an_overhanging_outline_onto_consistent_image_edges():
    roof=box(9,9,31,31)
    aligned,report=align_roof_faces([roof],aerial(),(0,0,40,40))
    expected=box(10,10,30,30)
    assert report['applied']
    assert report['scale']==pytest.approx(20/22,abs=.015)
    assert aligned[0].hausdorff_distance(expected)<.25


def test_does_not_shrink_towards_a_single_internal_roof_edge():
    data=np.full((400,400,3),30,dtype=np.uint8)
    data[100:300,100:300]=(170,100,65)
    data[195:205,100:300]=(245,245,245)
    roof=box(10,10,30,30)
    aligned,report=align_roof_faces([roof],Image.fromarray(data),(0,0,40,40))
    assert not report['applied']
    assert aligned[0].equals(roof)


def test_already_aligned_outline_stays_fixed():
    roof=box(10,10,30,30)
    aligned,report=align_roof_faces([roof],aerial(),(0,0,40,40))
    assert not report['applied'] and aligned[0].equals(roof)


def test_uniform_image_cannot_trigger_an_arbitrary_shift():
    roof=box(9,12,29,32)
    aligned,report=align_roof_faces([roof],Image.new('RGB',(400,400)),(0,0,40,40))
    assert not report['applied'] and aligned[0].equals(roof)
