from fastapi.testclient import TestClient
from PIL import Image
from shapely.geometry import box,shape
from shapely.ops import transform,unary_union
from app import main
from app.geodata import TO_SWISS

def setup_analysis(monkeypatch,detections):
    roof=box(2640000,1230000,2640012,1230012)
    record=dict(featureId=1,geometry={'rings':[list(roof.exterior.coords)]},
                attributes=dict(building_id=1,neigung=20,ausrichtung=0,mstrahlung=1000))
    monkeypatch.setattr(main,'status',lambda:{n:{'state':'ready','available':True} for n in ['swiss','rid','roof']})
    monkeypatch.setattr(main,'get_roofs',lambda *args:[record])
    monkeypatch.setattr(main,'get_image',lambda *args:(Image.new('RGB',(1000,1000)),roof.bounds))
    monkeypatch.setattr(main,'predict',lambda *args:detections)
    return TestClient(main.app)

def test_layout_requires_agreement_with_image_roof_boundary(monkeypatch):
    boundary=box(2640002,1230002,2640010,1230010)
    obstacle=box(2640004,1230004,2640006,1230006)
    detections=[dict(geometry=boundary,label='roof',confidence=.8,model='roof'),
                dict(geometry=obstacle,label='chimney',confidence=.8,model='rid')]
    client=setup_analysis(monkeypatch,detections)
    response=client.get('/api/analyze',params={'lat':47.2,'lon':7.9})
    assert response.status_code==200
    panels=[transform(TO_SWISS.transform,shape(f['geometry'])) for f in response.json()['geojson']['features'] if f['properties']['kind']=='panel']
    assert panels
    assert all(boundary.buffer(.001).covers(p) and not obstacle.intersects(p) for p in panels)

def test_no_detected_roof_means_no_panel_placement(monkeypatch):
    client=setup_analysis(monkeypatch,[])
    response=client.get('/api/analyze',params={'lat':47.2,'lon':7.9})
    assert response.status_code==200
    assert response.json()['panel_count']==0


def test_existing_pv_and_roof_objects_both_block_new_panels(monkeypatch):
    roof=box(2640000,1230000,2640012,1230012)
    installed=box(2640001,1230001,2640005,1230006)
    chimney=box(2640007,1230007,2640009,1230009)
    detections=[dict(geometry=roof,label='roof',confidence=.9,model='roof'),
                dict(geometry=installed,label='pv_installation',confidence=.9,model='swiss'),
                dict(geometry=chimney,label='chimney',confidence=.8,model='rid')]
    response=setup_analysis(monkeypatch,detections).get('/api/analyze',params={'lat':47.2,'lon':7.9})
    assert response.status_code==200
    features=response.json()['geojson']['features']
    assert any(f['properties']['kind']=='pv' for f in features)
    assert any(f['properties']['kind']=='obstacle' for f in features)
    panels=[transform(TO_SWISS.transform,shape(f['geometry'])) for f in features if f['properties']['kind']=='panel']
    assert panels
    exclusions=unary_union([installed,chimney])
    assert all(not exclusions.intersects(p) for p in panels)
    assert abs(sum(p.area for p in panels)-unary_union(panels).area)<.001
