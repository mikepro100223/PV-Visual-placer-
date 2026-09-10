"""Abbas obstacle evidence without replacing Yucan roof planes or PV."""
import asyncio
import numpy as np
import httpx
from shapely.ops import unary_union
from app.detections import normalise,pixel_to_world,world_to_pixel,polygon_parts
from app.obstacles import elevation,rooflights,geneva


async def height_obstacles(client,faces,bounds):
    left,bottom,right,top=bounds
    window=(left-2,bottom-2,right+2,top+2)
    hrefs=await elevation.tile_hrefs(client,window)
    if not hrefs or len(hrefs)>elevation.MAX_TILES:
        raise elevation.ElevationUnavailable('No bounded surface-height coverage')
    paths=[await elevation.cached_tile(client,href) for href in hrefs]
    heights,x,y=elevation.mosaic(paths,window)
    if np.count_nonzero(np.isfinite(heights))/heights.size<.95:
        raise elevation.ElevationUnavailable('Surface-height coverage is incomplete')
    return elevation.detect(heights,x,y,faces)


async def measured_obstacles(faces,bounds):
    detected=[];warnings=[]
    coverage={'height':'unavailable','geneva':'outside_coverage'}
    async with httpx.AsyncClient(timeout=30,follow_redirects=True) as client:
        try:
            for obj in await asyncio.wait_for(height_obstacles(client,faces,bounds),timeout=75):
                detected.append({**obj,'confidence':None,'source':'abbas_height','model':'abbas_height'})
            coverage['height']='ready'
        except (httpx.HTTPError,elevation.ElevationUnavailable,OSError,ValueError,TimeoutError) as exc:
            warnings.append(f'Height-based obstacle detection unavailable: {exc}. Image detections may miss raised structures.')
        e=geneva.EXTENT
        inside=not (bounds[2]<e[0] or bounds[0]>e[2] or bounds[3]<e[1] or bounds[1]>e[3])
        if inside:
            try:
                features=await asyncio.wait_for(geneva.superstructures(client,bounds),timeout=25)
                for polygon,props in geneva.surveyed_polygons(features,unary_union(faces)):
                    detected.append(dict(geometry=polygon,kind='other_obstacle',confidence=None,
                                         source='abbas_geneva_survey',model='abbas_geneva_survey',
                                         survey_date=props.get('DATE_LEVE')))
                coverage['geneva']='ready'
            except (httpx.HTTPError,ValueError,KeyError,TimeoutError) as exc:
                coverage['geneva']='unavailable'
                warnings.append(f'Geneva surveyed obstacles unavailable: {exc}')
    return normalise(detected),warnings,coverage


def detect_obstacles(image,bounds,faces,existing):
    detected,warnings,coverage=asyncio.run(measured_obstacles(faces,bounds))
    # Exclude Yucan PV first: blue modules must not become roof windows.
    roof=unary_union(faces)
    pixel_roof=world_to_pixel(roof,bounds,image.size)
    exclude=[part for obj in existing+detected
             for part in polygon_parts(world_to_pixel(obj['geometry'],bounds,image.size))]
    ppm=image.width/(bounds[2]-bounds[0])
    for obj in rooflights.detect(np.asarray(image),pixel_roof,ppm,exclude):
        geometry=pixel_to_world(obj['geometry'],bounds,image.size).intersection(roof)
        detected.append(dict(geometry=geometry,kind='skylight',confidence=None,
                             source='abbas_rooflight',model='abbas_rooflight'))
    coverage['rooflights']='ready'
    return normalise(detected),warnings,coverage
