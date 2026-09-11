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
NON_BLOCKING_KINDS={'shadow'}
SOURCE_PRIORITY={
    'abbas_geneva_survey': 50,
    'abbas_height': 40,
    'abbas_rooflight': 30,
    'swiss': 20,
    'rid': 10,
}


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


def _source(item):
    return item.get('source',item.get('model','unknown'))


def _representative(first,second):
    def rank(item):
        return (SOURCE_PRIORITY.get(_source(item),0),item.get('confidence') or 0)
    return max((first,second),key=rank)


def merge_detections(objects):
    """Stitch overlapping tile masks and retain deterministic provenance."""
    merged=[]
    for item in sorted(normalise(objects),key=lambda o:-(o['confidence'] or 0)):
        geometry=item['geometry']
        hits=[i for i,old in enumerate(merged) if old['kind']==item['kind']
              and geometry.intersection(old['geometry']).area/min(geometry.area,old['geometry'].area)>.5]
        if not hits:
            merged.append({**item,'source':_source(item),'sources':[_source(item)]})
            continue
        matches=[merged[i] for i in hits]
        representative=item
        for old in matches:representative=_representative(representative,old)
        sources={_source(item)}
        for old in matches:sources.update(old.get('sources',[_source(old)]))
        scores=[o['confidence'] for o in [item]+matches if o['confidence'] is not None]
        combined=unary_union([geometry]+[old['geometry'] for old in matches])
        merged=[old for i,old in enumerate(merged) if i not in hits]
        parts=list(polygon_parts(combined))
        for part in parts:
            merged.append({**representative,'geometry':part,'source':_source(representative),
                           'model':_source(representative),'sources':sorted(sources),
                           'confidence':max(scores) if scores else None})
    return merged


def blocks_placement(item):
    return item.get('kind',item.get('label')) not in NON_BLOCKING_KINDS


def arbitrate_detections(objects, trusted_pv_confidence=.5):
    """Resolve cross-model conflicts while preserving measured evidence.

    The dedicated Swiss PV model is substantially more precise than RID on the
    available held-out evaluations. Its confident mask therefore wins pixels
    which RID simultaneously labels as a physical obstacle. Survey, elevation
    and rooflight evidence is never removed by this model-to-model rule.
    """
    merged=merge_detections(objects)
    trusted=[]
    for item in merged:
        sources=set(item.get('sources',[_source(item)]))
        if (item['kind']=='pv_installation' and item.get('confidence') is not None
                and item['confidence']>=trusted_pv_confidence and 'swiss' in sources):
            trusted.append(item['geometry'])
    pv_union=unary_union(trusted) if trusted else Polygon()
    result=[]
    for item in merged:
        sources=set(item.get('sources',[_source(item)]))
        geometry=item['geometry']
        if item['kind'] not in {'pv_installation','roof','shadow'} and sources=={'rid'}:
            geometry=geometry.difference(pv_union)
        for part in polygon_parts(make_valid(geometry)):
            if part.area<=1e-8:continue
            result.append({**item,'geometry':part,'blocks_placement':blocks_placement(item)})
    return result


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
                blocks_placement=blocks_placement(item),
                crs='EPSG:2056')
