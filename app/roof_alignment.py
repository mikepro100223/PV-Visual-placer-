"""Conservative projection of official roof faces onto the analysis image.

Never transform image detections. Accept only a strong, unambiguous boundary
match within four metres. A small uniform shrink is allowed when all sides of
an official outline consistently overhang the visible roof; otherwise the
official coordinates are preserved.
"""
import math
import cv2
import numpy as np
from shapely import affinity
from shapely.ops import unary_union
from app.detections import world_to_pixel,polygon_parts


def align_roof_faces(faces,image,bounds,max_shift_m=4):
    report=dict(applied=False,east_m=0.0,north_m=0.0,scale=1.0,
                reason='insufficient_boundary_evidence')
    h,w=image.height,image.width
    sx,sy=(bounds[2]-bounds[0])/w,(bounds[3]-bounds[1])/h
    pixel=world_to_pixel(unary_union(faces),bounds,image.size)
    points=[];normals=[]
    for part in polygon_parts(pixel):
        coords=list(part.exterior.coords)
        for a,b in zip(coords,coords[1:]):
            a,b=np.asarray(a),np.asarray(b)
            vector=b-a;length=np.linalg.norm(vector)
            if length<10:continue
            normal=np.array([-vector[1],vector[0]])/length
            for distance in np.arange(5,length-5,3):
                points.append(a+vector*distance/length);normals.append(normal)
    if len(points)<60:return faces,report
    points=np.array(points);normals=np.array(normals)
    step=max(1,len(points)//1600);points=points[::step];normals=normals[::step]
    radius_x,radius_y=math.ceil(max_shift_m/sx),math.ceil(max_shift_m/sy)
    inside=(points[:,0]>=radius_x)&(points[:,0]<w-radius_x)&(points[:,1]>=radius_y)&(points[:,1]<h-radius_y)
    points,normals=points[inside],normals[inside]
    if len(points)<60:return faces,report
    gray=cv2.GaussianBlur(cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY),(5,5),0)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    centre=np.asarray(pixel.centroid.coords[0])
    def response(candidate_points,dx,dy):
        xy=np.rint(candidate_points+[dx,dy]).astype(int);x,y=xy[:,0],xy[:,1]
        return np.minimum(np.abs(gx[y,x]*normals[:,0]+gy[y,x]*normals[:,1]),100)
    baseline=response(points,0,0);base=float(baseline.mean())
    scores=[]
    # Scaling only shrinks: the reported failure is an outline overhanging the
    # visible roof. Expanding a registered roof from image texture is unsafe.
    for scale in np.linspace(.90,1.0,11):
        candidate=centre+(points-centre)*scale
        for dy in range(-radius_y,radius_y+1):
            for dx in range(-radius_x,radius_x+1):
                if math.hypot(dx*sx,dy*sy)<=max_shift_m:
                    scores.append((float(response(candidate,dx,dy).mean()),dx,dy,float(scale)))
    best,dx,dy,scale=max(scores)
    report.update(edge_score_before=round(base,2),edge_score_after=round(best,2))
    if best<30 or best<base*1.5 or best-base<12:
        report['reason']='no_clear_improvement';return faces,report
    # Similar peaks well away from the winning translation imply ambiguity.
    alternatives=[score for score,x,y,s in scores
                  if math.hypot((x-dx)*sx,(y-dy)*sy)>.75 or abs(s-scale)>.025]
    if alternatives and best<max(alternatives)*1.12:
        report['reason']='ambiguous_boundary';return faces,report
    if math.hypot(dx*sx,dy*sy)>max_shift_m-.25:
        report['reason']='search_limit';return faces,report
    improved=response(centre+(points-centre)*scale,dx,dy)
    groups=(np.abs(normals[:,1])>np.abs(normals[:,0])).astype(int)*2
    groups+=(np.where(groups==0,normals[:,0],normals[:,1])>0).astype(int)
    support=sum(np.count_nonzero(groups==g)>=10 and
                improved[groups==g].mean()>baseline[groups==g].mean()+8 for g in range(4))
    if support<3:
        report['reason']='insufficient_directional_support';return faces,report
    east,north=float(dx*sx),float(-dy*sy)
    report.update(applied=True,east_m=east,north_m=north,scale=scale,
                  reason='image_boundary_match')
    origin=unary_union(faces).centroid.coords[0]
    projected=[affinity.scale(face,xfact=scale,yfact=scale,origin=origin) for face in faces]
    return [affinity.translate(face,xoff=east,yoff=north) for face in projected],report
