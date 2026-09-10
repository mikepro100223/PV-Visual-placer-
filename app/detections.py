"""Shared polygon contract in LV95 (EPSG:2056).

Merging adapts Abbas' pv_segmentation without losing holes or repaired parts.
"""
import math
from shapely import make_valid, affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

ALIASES={'existing_pv':'pv_installation','solar_panel':'pv_installation',
         'solar_panels':'pv_installation','pv':'pv_installation',
         'unknown_obstacle':'other_obstacle','window':'skylight'}
KINDS={'pv_installation','roof','chimney','skylight','dormer','other_obstacle','ladder','tree','shadow'}


def polygon_parts(geometry):
    if geometry.is_empty:return
    if geometry.geom_type=='Polygon':yield geometry
    else:
        for part in getattr(geometry,'geoms',[]):yield from polygon_parts(part)


def normalise(objects):
    result=[]
    for item in objects:
        raw=item.get('kind',item.get('label'))
        kind=ALIASES.get(raw,raw)
        if kind not in KINDS:raise ValueError(f'Unsupported rooftop class: {kind}')
        score=item.get('confidence')
        if score is not None and (not math.isfinite(score) or not 0<=score<=1):
            raise ValueError('Detection confidence must be between 0 and 1')
        geometry=item.get('geometry')
        if geometry is None:geometry=Polygon(item['polygon'],item.get('holes',[]))
        for part in polygon_parts(make_valid(geometry)):
            if part.area<=1e-8:continue
            result.append({**item,'kind':kind,'label':kind,'geometry':part,
                           'confidence':float(score) if score is not None else None,
                           'model':item.get('model',item.get('source','unknown'))})
    return result


def merge_detections(objects):
    merged=[]
    for item in sorted(normalise(objects),key=lambda o:-(o['confidence'] or 0)):
        geometry=item['geometry']
        hits=[i for i,old in enumerate(merged) if old['kind']==item['kind']
              and geometry.intersection(old['geometry']).area/min(geometry.area,old['geometry'].area)>.5]
        sources={item.get('source',item['model'])}
        for i in hits:sources.update(merged[i]['sources'])
        scores=[o['confidence'] for o in [item]+[merged[i] for i in hits] if o['confidence'] is not None]
        geometry=unary_union([geometry]+[merged[i]['geometry'] for i in hits])
        merged=[old for i,old in enumerate(merged) if i not in hits]
        for part in polygon_parts(geometry):
            merged.append({**item,'geometry':part,'sources':sorted(sources),'confidence':max(scores) if scores else None})
    return merged


def pixel_to_world(geometry,bounds,size,offset=(0,0)):
    left,bottom,right,top=bounds
    sx,sy=(right-left)/size[0],(top-bottom)/size[1]
    return affinity.affine_transform(geometry,[sx,0,0,-sy,left+offset[0]*sx,top-offset[1]*sy]).intersection(box(*bounds))


def world_to_pixel(geometry,bounds,size):
    left,bottom,right,top=bounds
    sx,sy=size[0]/(right-left),size[1]/(top-bottom)
    return affinity.affine_transform(geometry,[sx,0,0,-sy,-left*sx,top*sy])


def public_detection(item):
    p=item['geometry']
    return dict(kind=item['kind'],confidence=item['confidence'],
                polygon=[list(xy) for xy in p.exterior.coords[:-1]],
                holes=[[list(xy) for xy in ring.coords[:-1]] for ring in p.interiors],
                source=item.get('source',item['model']),sources=item.get('sources',[item['model']]),
                crs='EPSG:2056')
