import math
import io
import json
import pytest
from PIL import Image
from pydantic import ValidationError
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas.analysis import AnalysisSettings, MarkedObject


def detailed_array(count=406):
    return [[100+(30+3*math.sin(9*i*2*math.pi/count))*math.cos(i*2*math.pi/count),
             100+(30+3*math.sin(9*i*2*math.pi/count))*math.sin(i*2*math.pi/count)] for i in range(count)]


def test_406_vertex_recognition_mask_round_trips_unchanged():
    polygon = detailed_array()
    settings = AnalysisSettings(roof=[(0,0),(200,0),(200,200),(0,200)],
        objects=[{"polygon":polygon,"kind":"existing_pv","source":"image"}])
    assert len(settings.objects[0].polygon) == 406
    assert settings.model_dump(mode='json')['objects'][0]['polygon'] == polygon


def test_detailed_recognition_mask_is_accepted_by_analysis_endpoint():
    image=Image.new('RGB',(200,200),'gray');stream=io.BytesIO();image.save(stream,format='PNG')
    settings={"roof":[[0,0],[200,0],[200,200],[0,200]],"pixels_per_metre":10,
        "use_ai":False,"scale_verified":True,
        "objects":[{"polygon":detailed_array(),"kind":"existing_pv","source":"image"}]}
    response=TestClient(app).post('/api/analyse',files={'image':('roof.png',stream.getvalue(),'image/png')},
                                 data={'settings':json.dumps(settings)})
    assert response.status_code == 200, response.text
    assert response.json()['existing_pv']


def test_roof_edit_limit_and_invalid_coordinates_are_still_checked():
    with pytest.raises(ValidationError):
        AnalysisSettings(roof=detailed_array())
    with pytest.raises(ValidationError):
        MarkedObject(polygon=detailed_array(4097))
    with pytest.raises(ValidationError, match='finite'):
        AnalysisSettings(roof=[(0,0),(200,0),(200,200)],objects=[{
            "polygon":[(0,0),(1,1),(float('nan'),2)]}])
