"""Join held-out SWISSIMAGE chips to Sonnendach and export surface-metric free area.

Survey-reference errors concern catalogued superstructures only, not exhaustive
occupancy. No shade, structural suitability or capacity accuracy is implied.
"""
import argparse
import asyncio
import json
from pathlib import Path
import httpx
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform, unary_union
from ultralytics import YOLO
from backend.services.map_service import feature_planes, API, ROOF_LAYER, TO_WGS84
from backend.services.roof_plane import official_plane


def image_to_world(geometry, bbox):
    return transform(lambda x,y: (np.asarray(x)*.1+bbox[0], bbox[3]-np.asarray(y)*.1), geometry)


def mask_geometry(model, image):
    result = model.predict(image, imgsz=640, conf=.25, retina_masks=True, verbose=False)[0]
    return unary_union([Polygon(p).buffer(0) for p in result.masks.xy if len(p)>=3]) if result.masks is not None else Polygon()


async def evaluate(args):
    root = Path(args.data)
    records = [r for r in json.loads((root/'split_manifest.json').read_text()) if r['split']=='test'][:args.chips]
    obstacle_model = YOLO(args.obstacle_model) if args.obstacle_source == 'model' else None
    pv_model = YOLO(args.pv_model)
    features, rows, skipped = [], [], []
    async with httpx.AsyncClient(timeout=40) as client:
        for record in records:
            bbox, name = record['bbox'], record['source']
            image = Image.open(root/'images/test'/f'{name}.jpg').convert('RGB')
            obstacles = image_to_world(mask_geometry(obstacle_model, image), bbox) if obstacle_model else None
            pv = image_to_world(mask_geometry(pv_model, image), bbox)
            truth = []
            for line in (root/'labels/test'/f'{name}.txt').read_text().splitlines():
                points = np.array(list(map(float,line.split()[1:]))).reshape(-1,2)*640
                truth.append(image_to_world(Polygon(points),bbox))
            truth = unary_union(truth)
            if obstacles is None:
                obstacles = truth
            cache = root/f'{name}-sonnendach.json'
            if cache.exists():
                data = json.loads(cache.read_text())
            else:
                response = await client.get(API+'/MapServer/identify',params={
                    'geometry':','.join(map(str,bbox)), 'geometryType':'esriGeometryEnvelope',
                    'layers':'all:'+ROOF_LAYER, 'sr':2056, 'geometryFormat':'geojson',
                    'returnGeometry':'true','tolerance':0,'mapExtent':','.join(map(str,bbox)),
                    'imageDisplay':'640,640,96','limit':1000})
                response.raise_for_status();data=response.json()
                if 'error' in data:
                    raise ValueError('Sonnendach lookup failed during evaluation')
                cache.write_text(json.dumps(data))
            chip = box(*bbox)
            planes = feature_planes(data.get('results',[]),chip.centroid)
            for face in planes:
                roof = face['geometry']
                if not chip.covers(roof) or roof.area < 5:
                    skipped.append(face['id']);continue
                plane = official_plane(roof,face['properties'])
                if plane.source == 'projected_2d':
                    skipped.append(face['id']);continue
                local = plane.local_geometry(roof)
                detected = plane.local_geometry(obstacles.intersection(roof))
                reference = plane.local_geometry(truth.intersection(roof))
                existing = plane.local_geometry(pv.intersection(roof))
                clear = local.buffer(-.3).difference(detected.buffer(.4))
                reference_clear = local.buffer(-.3).difference(reference.buffer(.4))
                usable = clear.difference(existing.buffer(.2))
                union = detected.union(reference).area
                row = {'roof_id':face['id'], 'chip':name, 'pitch_deg':plane.describe()['pitch_deg'],
                    'projected_area_m2':roof.area, 'roof_surface_area_m2':local.area,
                    'detected_obstacle_area_m2':detected.area,'survey_obstacle_area_m2':reference.area,
                    'detected_pv_area_m2':existing.area,'usable_surface_area_m2':usable.area,
                    'survey_reference_clear_area_error_m2':clear.area-reference_clear.area if obstacle_model else None,
                    'survey_obstacle_iou':detected.intersection(reference).area/union if union and obstacle_model else None}
                rows.append(row)
                world = plane.world_geometry(usable)
                features.append({'type':'Feature','geometry':mapping(transform(TO_WGS84.transform,world)),
                    'properties':row})
            print(f"Mapped {name}; {len(rows)} complete roof faces",flush=True)
    report = {'chips':len(records),'complete_faces':len(rows),'skipped_partial_or_unmeasured_faces':len(skipped),
        'survey_reference_clear_area_mae_m2':float(np.mean([abs(r['survey_reference_clear_area_error_m2']) for r in rows])) if rows and obstacle_model else None,
        'obstacle_source':args.obstacle_source,
        'scope':'Surface-metre usable areas after predicted PV and survey/model obstacles with 0.3 m edge / 0.4 m obstacle / 0.2 m PV clearances. Before shade and structural checks. Survey mode is a reference calculation, not an accuracy evaluation.',
        'limitations':'Catalogue is incomplete and can be displaced from imagery. Reference errors are not exhaustive usable-area or capacity accuracy. Complete roof faces only; building siblings can extend outside the chip.',
        'roofs':rows}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    Path(args.output).with_suffix('.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='roofs'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/geneva_clean')
    parser.add_argument('--obstacle-model',default='models/obstacle_best.pt')
    parser.add_argument('--obstacle-source',choices=['survey','model'],default='survey')
    parser.add_argument('--pv-model',default='models/rooftop_best.pt')
    parser.add_argument('--chips',type=int,default=12)
    parser.add_argument('--output',default='models/geneva-roof-evaluation.json')
    asyncio.run(evaluate(parser.parse_args()))
