"""Image evidence for repeated fan banks absent from an older height survey.

This is a geometric image heuristic, not a trained obstacle classifier. Require
several similarly sized dark circular fans in a regular cluster on a brighter
housing, exclude known PV, and keep the housing rather than individual blades.
"""
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from .pv_field_service import _mask_from


def detect(image, roof, pixels_per_metre, existing_pv=None):
    ppm = pixels_per_metre
    if ppm <= 0 or roof.is_empty:
        return []
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    allowed = _mask_from(roof, gray.shape).astype(bool)
    pv = unary_union(existing_pv or [])
    if not pv.is_empty:
        allowed &= ~_mask_from(pv, gray.shape).astype(bool)
    span = max(5, round(1.5*ppm) | 1)
    contrast = cv2.GaussianBlur(gray, (span,span), 0)-gray
    dark = ((contrast > 22) & (gray < 150) & allowed).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(dark,8)
    fans = []
    for index in range(1,count):
        x,y,w,h,pixels = stats[index]
        if not (.025 <= pixels/ppm**2 <= .65 and .45 <= w/max(h,1) <= 2.2
                and .2 <= min(w,h)/ppm and max(w,h)/ppm <= 1.4):
            continue
        component = (labels[y:y+h,x:x+w] == index).astype(np.uint8)
        contours,_ = cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours,key=cv2.contourArea)
        perimeter = cv2.arcLength(contour,True)
        if perimeter <= 0 or 4*np.pi*cv2.contourArea(contour)/perimeter**2 < .42:
            continue
        pad=max(2,round(.35*ppm))
        around=gray[max(0,y-pad):min(gray.shape[0],y+h+pad),max(0,x-pad):min(gray.shape[1],x+w+pad)]
        if np.percentile(around,65) < 145:
            continue
        fans.append((Point(*centres[index]),float(pixels)))
    pending=set(range(len(fans)));found=[]
    while pending:
        group=[pending.pop()]
        for i in group:
            close=[j for j in pending if fans[i][0].distance(fans[j][0]) <= 1.7*ppm]
            group.extend(close);pending.difference_update(close)
        if len(group) < 6:
            continue
        areas=np.array([fans[i][1] for i in group])
        if np.std(areas)/np.mean(areas) > .55:
            continue
        centres_in_group=[fans[i][0] for i in group]
        nearest=np.array([min(p.distance(q) for j,q in enumerate(centres_in_group) if j!=i)
                          for i,p in enumerate(centres_in_group)])
        if np.std(nearest)/np.mean(nearest) > .35:
            continue
        housing=unary_union(centres_in_group).minimum_rotated_rectangle.buffer(.65*ppm,join_style=2)
        if housing.geom_type != 'Polygon' or not 3 <= housing.area/ppm**2 <= 150:
            continue
        if housing.intersection(pv).area > .05*housing.area:
            continue
        housing=housing.intersection(roof)
        if housing.geom_type == 'Polygon' and not housing.is_empty:
            found.append({'geometry':housing,'fan_count':len(group),'source':'image'})
    return found
