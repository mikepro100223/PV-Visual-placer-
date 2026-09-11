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
from app.geometry import MODULE, esri_polygon, pack_building
from app.inference import status, predict
from app.detections import arbitrate_detections,blocks_placement,normalise,public_detection
from app.obstacles.pipeline import detect_obstacles
from app.roof_alignment import align_roof_faces

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
            setback: float=Query(default=.6,ge=.3,le=2),
            row_gap: float=Query(default=.35,ge=.2,le=2),
            obstacle_confidence: float=Query(default=.30,ge=.1,le=.9)):
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
        detections = normalise(predict(image,bounds,confidence,obstacle_confidence))
        roof_predictions=[d['geometry'] for d in detections if d['label']=='roof']
        detected_roof=unary_union(roof_predictions) if roof_predictions else None
        extra,obstacle_warnings,obstacle_sources=detect_obstacles(
            image,bounds,geometries,[d for d in detections if d['label']!='roof'])
        # Align the roof to this exact image once. All image/model and measured
        # obstacle coordinates stay fixed; display and placement share the result.
        geometries,roof_alignment=align_roof_faces(geometries,image,bounds)
        whole=unary_union(geometries)
        detections.extend(extra)
        detections = [dict(d,geometry=d['geometry'].intersection(whole)) for d in detections if d['label']!='roof' and d['geometry'].intersects(whole)]
        detections = [d for d in detections if not d['geometry'].is_empty and d['geometry'].area>.01]
        detections=arbitrate_detections(detections)
        blocking=[d for d in detections if blocks_placement(d)]
        features, facets = [], []
        panel_count,total_usable,total_energy = 0,0,0
        warnings = list(obstacle_warnings)
        if not all(m['available'] for m in available.values()):
            warnings.append('Some roof models are unavailable. This layout is provisional; obstacles or roof-boundary errors may be missed.')
        planning_faces=[]
        for record,roof in zip(records,geometries):
            attrs=record['attributes']
            valid=attrs.get('neigung') is not None and attrs.get('ausrichtung') is not None and 0<=float(attrs['neigung'])<80
            # When roof segmentation is available, require agreement between
            # image-based roof detection and the geodata boundary.
            planning_roof=roof.intersection(detected_roof) if detected_roof is not None else roof
            if available.get('roof',{}).get('available') and detected_roof is None:
                planning_roof=roof.difference(roof)
            planning_faces.append(dict(geometry=planning_roof if valid else roof.difference(roof),
                                       slope=float(attrs['neigung']) if valid else 0,
                                       azimuth=(float(attrs['ausrichtung'])+180)%360 if valid else 0))
        layouts=pack_building(planning_faces,[d['geometry'] for d in blocking],setback=setback,row_gap=row_gap)
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
            panels,stats = layouts[i]
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
            display_roof=planning_faces[i]['geometry']
            if i:
                display_roof=display_roof.difference(unary_union([f['geometry'] for f in planning_faces[:i]]))
            if not display_roof.is_empty:
                features.append(feature(display_roof,kind='roof',pitch=slope,azimuth=azimuth,facet=i))
            free=planning_faces[i]['geometry'].difference(unary_union([d['geometry'] for d in blocking]))
            if not free.is_empty:
                features.append(feature(free,kind='free',label='Roof without detected PV or obstacles',facet=i))
            features.extend(feature(p,kind='panel',facet=i,power_w=450) for p in panels)
        for d in detections:
            features.append(feature(d['geometry'],kind='pv' if d['label']=='pv_installation' else 'obstacle',
                                    label=d['label'],confidence=d['confidence'],model=d['model'],
                                    source=d.get('source',d['model']),sources=d.get('sources',[]),
                                    blocks_placement=blocks_placement(d)))
        if any(d['label']=='shadow' for d in detections):
            warnings.append('Visible image shadows are advisory and do not block placement; seasonal shadow movement is not simulated.')
        if any(f['pitch_deg']<1 for f in facets):
            warnings.append('Flat roofs reserve at least 1 m between rows. Exact rack tilt and seasonal self-shading still need installation design.')
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
                    provisional=not all(m['available'] and m.get('state')=='ready' for m in available.values())
                                or 'unavailable' in obstacle_sources.values(),
                    detections=[public_detection(d) for d in detections],obstacle_sources=obstacle_sources,
                    roof_alignment=roof_alignment,
                    spacing=dict(edge_m=setback,obstacle_m=.5,module_gap_m=.1,row_gap_m=row_gap,flat_row_gap_m=max(1,row_gap),access_aisle_m=.8),
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
