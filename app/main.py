from dataclasses import asdict
import io
import base64
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import mapping
from shapely.ops import transform, unary_union
import requests

from app.geodata import TO_WGS, get_roofs, get_image, search_address
from app.geometry import MODULE, esri_polygon, pack_panels
from app.inference import status, predict

app = FastAPI(title='PV Visual Placer',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET'])
ANALYSIS_LOCK = threading.Lock()

@app.get('/api/status')
def model_status():
    return dict(models=status(),module=asdict(MODULE))

@app.get('/api/search')
def search(q: str=Query(min_length=3,max_length=150)):
    try:
        return search_address(q)
    except (requests.RequestException,ValueError) as exc:
        raise HTTPException(502,'Swiss address service unavailable. Try clicking the map.') from exc

def feature(geometry, **properties):
    return dict(type='Feature',geometry=mapping(transform(TO_WGS.transform,geometry)),properties=properties)

@app.get('/api/analyze')
def analyze(lat: float=Query(ge=45.7,le=47.9),lon: float=Query(ge=5.9,le=10.6),
            confidence: float=Query(default=.25,ge=.1,le=.9),
            setback: float=Query(default=.3,ge=0,le=2)):
    available = status()
    if not any(m['available'] for m in available.values()):
        raise HTTPException(503,'The roof models are still training. Watch the model status and try again when a checkpoint is ready.')
    if not ANALYSIS_LOCK.acquire(blocking=False):
        raise HTTPException(429,'Another roof is being analyzed. Please wait a moment.')
    try:
        records = get_roofs(lat,lon)
        if not records:
            raise HTTPException(404,'No Sonnendach roof at this point. Click inside a roof, or try a nearby house.')
        geometries = [esri_polygon(r['geometry']) for r in records]
        whole = unary_union(geometries)
        image,bounds = get_image(whole.bounds)
        detections = predict(image,bounds,confidence)
        detections = [dict(d,geometry=d['geometry'].intersection(whole)) for d in detections if d['geometry'].intersects(whole)]
        detections = [d for d in detections if not d['geometry'].is_empty and d['geometry'].area>.01]
        features, facets = [], []
        panel_count,total_usable,total_energy = 0,0,0
        warnings = []
        if not all(m['available'] for m in available.values()):
            warnings.append('Only one model is ready. Existing PV or obstacles may be missed.')
        for i,(record,roof) in enumerate(zip(records,geometries)):
            attrs = record['attributes']
            if attrs.get('neigung') is None or attrs.get('ausrichtung') is None:
                warnings.append('A roof facet has no pitch or direction; no panels placed on it.')
                continue
            slope = float(attrs['neigung'])
            azimuth = (float(attrs['ausrichtung'])+180)%360
            if slope>=80:
                warnings.append('Very steep roof facet excluded.')
                continue
            blockers = [d['geometry'] for d in detections if d['geometry'].intersects(roof)]
            panels,stats = pack_panels(roof,blockers,slope,azimuth,setback=setback)
            irradiance = attrs.get('mstrahlung')
            energy = len(panels)*MODULE.power_w/1000*float(irradiance)*.8 if irradiance is not None else None
            if energy is None:
                warnings.append('Annual radiation missing for a facet; its energy is not estimated.')
            else:
                total_energy += energy
            panel_count += len(panels)
            total_usable += stats['usable_area_m2']
            facets.append(dict(id=record['featureId'],pitch_deg=slope,azimuth_deg=azimuth,
                               panel_count=len(panels),kwp=round(len(panels)*.45,2),
                               annual_kwh=round(energy) if energy is not None else None,
                               irradiance_kwh_m2=irradiance,**stats))
            features.append(feature(roof,kind='roof',pitch=slope,azimuth=azimuth,facet=i))
            features.extend(feature(p,kind='panel',facet=i,power_w=450) for p in panels)
        for d in detections:
            features.append(feature(d['geometry'],kind='pv' if d['label']=='pv_installation' else 'obstacle',
                                    label=d['label'],confidence=d['confidence'],model=d['model']))
        if any(d['label']=='shadow' for d in detections):
            warnings.append('Visible image shadows are excluded conservatively; seasonal shadow movement is not simulated.')
        if any(f['pitch_deg']<1 for f in facets):
            warnings.append('Flat-roof layout assumes modules mounted flat. Tilted racks need extra row spacing.')
        warnings.extend(['Image predictions can miss small or obscured objects. Review the overlay.',
                         'Energy uses Sonnendach annual radiation and an assumed 80% performance ratio. No new hourly weather or 3D shadow simulation.',
                         'Clearances are prototype assumptions; roof condition, structural capacity and installation rules are unverified.'])
        preview = image.copy(); preview.thumbnail((500,500))
        buffer = io.BytesIO(); preview.save(buffer,format='JPEG')
        return dict(building_id=records[0]['attributes'].get('building_id'),
                    geojson=dict(type='FeatureCollection',features=features),facets=facets,
                    panel_count=panel_count,additional_kwp=round(panel_count*.45,2),
                    annual_kwh=round(total_energy) if all(f['annual_kwh'] is not None for f in facets) else None,
                    usable_area_m2=round(total_usable,1),
                    existing_pv_area_m2=round(unary_union([d['geometry'] for d in detections if d['label']=='pv_installation']).area,1),
                    module=asdict(MODULE),warnings=list(dict.fromkeys(warnings)),
                    models=available,image='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode())
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(502,'Swiss map service could not be reached. Try again shortly.') from exc
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    finally:
        ANALYSIS_LOCK.release()
