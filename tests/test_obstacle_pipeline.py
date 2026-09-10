import asyncio
import httpx
import numpy as np
from PIL import Image
from shapely.geometry import box,mapping
from app.obstacles import pipeline,geneva
from app.detections import normalise


def test_height_failure_is_reported_but_rooflight_detection_continues(monkeypatch):
    async def unavailable(*args):raise pipeline.elevation.ElevationUnavailable('missing tile')
    monkeypatch.setattr(pipeline,'height_obstacles',unavailable)
    image=np.full((200,200,3),(140,125,118),dtype=np.uint8)
    image[94:106,95:105]=(98,101,118)
    bounds=(2640000,1230000,2640020,1230020)
    found,warnings,coverage=pipeline.detect_obstacles(Image.fromarray(image),bounds,[box(*bounds)],[])
    assert coverage['height']=='unavailable' and warnings
    assert any(d['kind']=='skylight' and d['confidence'] is None for d in found)
    # The same blue patch is never reclassified as a window if PV already covers it.
    pv=normalise([dict(kind='existing_pv',geometry=box(2640008,1230008,2640012,1230012),confidence=.9)])
    assert pipeline.detect_obstacles(Image.fromarray(image),bounds,[box(*bounds)],pv)[0]==[]


def test_geneva_complete_pagination_and_clipping():
    offsets=[]
    def handler(request):
        offset=int(request.url.params['resultOffset']);offsets.append(offset)
        return httpx.Response(200,json={'crs':{'properties':{'name':'EPSG:2056'}},
            'features':[{'geometry':mapping(box(0,0,8,8)),'properties':{}}],
            'exceededTransferLimit':offset==0})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await geneva.superstructures(client,(2499000,1117000,2499100,1117100))
    features=asyncio.run(run())
    assert offsets==[0,4000] and len(features)==2
    assert list(geneva.surveyed_polygons(features,box(4,4,10,10)))[0][0].area==16
    assert asyncio.run(geneva.superstructures(None,(2600000,1200000,2600100,1200100)))==[]


def test_geneva_rejects_wrong_coordinate_system():
    import pytest
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,
                    json={'crs':{'properties':{'name':'EPSG:4326'}},'features':[]}))) as client:
            await geneva.superstructures(client,(2499000,1117000,2499100,1117100))
    with pytest.raises(ValueError,match='coordinate system'):asyncio.run(run())
