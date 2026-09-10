"""Public swisstopo imagery and Sonnendach roof lookup; local disk cache."""
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image
from pyproj import Transformer
import requests
from shapely.ops import unary_union

from app.geometry import esri_polygon

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/cache'
LAYER = 'ch.bfe.solarenergie-eignung-daecher'
API = 'https://api3.geo.admin.ch/rest/services/api'
TO_SWISS = Transformer.from_crs(4326,2056,always_xy=True)
TO_WGS = Transformer.from_crs(2056,4326,always_xy=True)

def cached_get(url, params, suffix):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((url+json.dumps(params,sort_keys=True)).encode()).hexdigest()
    path = CACHE / (key+suffix)
    if path.exists():
        return path.read_bytes()
    r = requests.get(url,params=params,timeout=(10,45))
    r.raise_for_status()
    # Validate before persisting an upstream error disguised as HTTP 200.
    if suffix == '.json':
        value = r.json()
        if 'error' in value:
            raise ValueError(str(value['error']))
    else:
        with Image.open(io.BytesIO(r.content)) as img:
            img.verify()
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(r.content)
    temporary.replace(path)
    return r.content

def connected_roof_parts(selected, nearby):
    """Join touching parts of the same registered house, not its neighbours."""
    egid=selected[0]['attributes'].get('gwr_egid')
    if not egid or int(egid)<=0:
        return selected
    accepted={r['featureId']:r for r in selected}
    footprint=unary_union([esri_polygon(r['geometry']) for r in selected])
    pending=[r for r in nearby if r.get('geometry',{}).get('rings')
             and r['attributes'].get('gwr_egid')==egid and r['featureId'] not in accepted]
    changed=True
    while changed:
        changed=False
        for record in pending[:]:
            geometry=esri_polygon(record['geometry'])
            if footprint.distance(geometry)>1.5:
                continue
            combined=footprint.union(geometry)
            left,bottom,right,top=combined.bounds
            if max(right-left,top-bottom)>168:
                continue
            accepted[record['featureId']]=record
            footprint=combined
            pending.remove(record)
            changed=True
    return list(accepted.values())


def get_roofs(lat, lon):
    x,y = TO_SWISS.transform(lon,lat)
    params = dict(geometryType='esriGeometryPoint', geometry=f'{x},{y}',
                  returnGeometry='true', layers='all:'+LAYER, tolerance=0,
                  sr=2056, lang='en', geometryFormat='esrijson')
    data = json.loads(cached_get(API+'/MapServer/identify',params,'.json'))
    results = data.get('results',[])
    if not results:
        return []
    building = results[0]['attributes'].get('building_id')
    if building is not None:
        params = dict(layer=LAYER,searchField='building_id',searchText=str(building),
                      contains='false',returnGeometry='true',sr=2056,lang='en')
        data = json.loads(cached_get(API+'/MapServer/find',params,'.json'))
        all_roofs = data.get('results',[])
        if all_roofs and all(r.get('geometry',{}).get('rings') for r in all_roofs):
            results=all_roofs
    # Sonnendach building_id can identify only a wing, entrance or dormer.
    # A bounded spatial query is faster than a national find on gwr_egid.
    if results[0]['attributes'].get('gwr_egid'):
        params=dict(geometryType='esriGeometryEnvelope',geometry=f'{x-70},{y-70},{x+70},{y+70}',
                    returnGeometry='true',layers='all:'+LAYER,tolerance=0,sr=2056,
                    lang='en',geometryFormat='esrijson')
        try:
            nearby=json.loads(cached_get(API+'/MapServer/identify',params,'.json')).get('results',[])
            results=connected_roof_parts(results,nearby)
        except (requests.RequestException,ValueError):
            pass  # Keep the selected part if the optional spatial query fails.
    return results

def get_image(bounds):
    minx,miny,maxx,maxy = bounds
    side = max(40, max(maxx-minx,maxy-miny)+12)
    if side > 180:
        raise ValueError('This building is too large for the prototype. Choose a smaller house.')
    cx,cy = (minx+maxx)/2,(miny+maxy)/2
    bounds = (cx-side/2,cy-side/2,cx+side/2,cy+side/2)
    size = min(1800, math.ceil(side/0.1))
    params = dict(SERVICE='WMS',VERSION='1.3.0',REQUEST='GetMap',
                  LAYERS='ch.swisstopo.swissimage',STYLES='',CRS='EPSG:2056',
                  BBOX=','.join(f'{v:.3f}' for v in bounds),WIDTH=size,HEIGHT=size,
                  FORMAT='image/jpeg')
    content = cached_get('https://wms.geo.admin.ch/', params, '.jpg')
    return Image.open(io.BytesIO(content)).convert('RGB'), bounds

def search_address(query):
    params = dict(searchText=query,type='locations',origins='address',limit=5,sr=4326)
    data = json.loads(cached_get(API+'/SearchServer',params,'.json'))
    return [dict(label=r['attrs']['label'],lat=r['attrs']['lat'],lon=r['attrs']['lon']) for r in data.get('results',[])]
