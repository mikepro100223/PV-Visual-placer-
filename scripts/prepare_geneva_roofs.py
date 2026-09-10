"""Add roof supervision from local Geneva TIFFs and SOLKAT_DACH.gpkg.

These are geodata-derived (weak) roof labels, not hand-reviewed obstacle masks.
No network calls. TIFF chips and masks retain the same 0.1 m pixel grid.
"""
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3

import cv2
import numpy as np
from PIL import Image
from shapely import from_wkb
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from prepare_swiss import mask_polygons,split_group

ROOT=Path(__file__).resolve().parents[1]

def gpkg_geometry(blob):
    if blob[:2]!=b'GP':raise ValueError('Not a GeoPackage geometry')
    envelope=(blob[3]>>1)&7
    size={0:0,1:32,2:48,3:48,4:64}.get(envelope)
    if size is None:raise ValueError('Invalid GeoPackage envelope')
    return from_wkb(blob[8+size:]).buffer(0)

def main():
    source=ROOT/'datasets';target=ROOT/'data/processed/roof'
    if not (target/'dataset.yaml').exists():
        raise SystemExit('Run prepare_swiss.py --source datasets/kaggle-swiss-solar-panels-segmentation --task roof first')
    c=sqlite3.connect((source/'SOLKAT_DACH.gpkg').resolve().as_uri()+'?mode=ro',uri=True)
    generator=random.Random(42);rows=[];errors=[];samples=[]
    tiffs=sorted((source/'swissimage-geneva').glob('*.tif'))
    # Verified local swisstopo 10,000 x 10,000 RGB tiles, processed one at a time.
    Image.MAX_IMAGE_PIXELS=120_000_000
    for index,path in enumerate(tiffs,1):
        try:
            with Image.open(path) as image:
                w,h=image.size;dx,dy=image.tag_v2[33550][:2];x,y=image.tag_v2[33922][3:5]
                if (w,h)!=(10000,10000) or abs(dx-.1)>1e-9 or abs(dy-.1)>1e-9:
                    raise ValueError('Unexpected source resolution')
                candidates=c.execute('SELECT a.geom FROM SOLKAT_CH_DACH a JOIN rtree_SOLKAT_CH_DACH_geom r ON r.id=a.objectid WHERE r.minx<? AND r.maxx>? AND r.miny<? AND r.maxy>?',(x+w*dx,x,y,y-h*dy))
                geoms=[gpkg_geometry(blob) for (blob,) in candidates]
                if not geoms:continue
                tree=STRtree(geoms);positions=[]
                for col in range(10):
                    for row in range(10):
                        xmin,ymax=x+col*100,y-row*100
                        chip_box=box(xmin,ymax-100,xmin+100,ymax)
                        indices=tree.query(chip_box,predicate='intersects')
                        clipped=unary_union([geoms[int(j)].intersection(chip_box) for j in indices])
                        if clipped.area>=100:
                            positions.append((col,row,clipped))
                generator.shuffle(positions)
                for col,row,geometry in positions[:8]:
                    xmin,ymax=x+col*100,y-row*100
                    rgb=image.crop((col*1000,row*1000,(col+1)*1000,(row+1)*1000)).convert('RGB')
                    mask=np.zeros((1000,1000),dtype=np.uint8)
                    polygons=list(geometry.geoms) if hasattr(geometry,'geoms') else [geometry]
                    for polygon in polygons:
                        if polygon.geom_type!='Polygon':continue
                        def pixel_ring(ring):
                            return np.array([((p[0]-xmin)/.1,(ymax-p[1])/.1) for p in ring.coords],dtype=np.int32)
                        cv2.fillPoly(mask,[pixel_ring(polygon.exterior)],1)
                        for hole in polygon.interiors:cv2.fillPoly(mask,[pixel_ring(hole)],0)
                    labels,iou=mask_polygons(mask,1,0)
                    if not labels:continue
                    group=f'{int(xmin)//2000}:{int(ymax-100)//2000}'
                    split=split_group(group);name=f'geneva_{int(xmin)}_{int(ymax-100)}'
                    for kind in ['images','labels']:(target/kind/split).mkdir(parents=True,exist_ok=True)
                    rgb.save(target/'images'/split/(name+'.jpg'),quality=95)
                    (target/'labels'/split/(name+'.txt')).write_text('\n'.join(labels))
                    rows.append(dict(file=name+'.jpg',split=split,group=group,source_tile=path.name,instances=len(labels),conversion_iou=iou,
                                     x=int(xmin),y=int(ymax-100),supervision='Sonnendach roof polygons; not obstacle labels'))
                    if len(samples)<12:
                        pixels=np.array(rgb);pixels[mask>0]=(pixels[mask>0]*.5+np.array([30,230,150])*.5).astype(np.uint8)
                        samples.append(Image.fromarray(pixels).resize((320,320)))
            print(f'{index}/{len(tiffs)} tiles, {len(rows)} roof chips',flush=True)
        except (OSError,ValueError) as exc:
            errors.append(dict(file=path.name,error=str(exc)))
    c.close()
    # Remove any new training chips that touch an existing held-out chip, and
    # vice versa. Manifests, not directory contents, define the training input.
    report_dir=ROOT/'data/reports'
    base=json.loads((report_dir/'roof_audit.json').read_text())
    combined=base['files']+rows
    excluded=set(base.get('excluded_train_boundary_files',[]))
    for target_split,higher_splits in [('val',{'test'}),('train',{'val','test'})]:
        held=[r for r in combined if r['split'] in higher_splits and r['file'] not in excluded]
        for r in combined:
            if r['split']==target_split and any(abs(r['x']-h['x'])<=100 and abs(r['y']-h['y'])<=100 for h in held):
                excluded.add(r['file'])
    for split in ['train','val','test']:
        accepted=[r for r in combined if r['split']==split and r['file'] not in excluded]
        (target/(split+'.txt')).write_text('\n'.join((target/'images'/split/r['file']).as_posix() for r in accepted))
    audit=dict(source_directory=str(source),geneva_tiles=len(tiffs),new_chips=len(rows),counts=dict(Counter(r['split'] for r in combined if r['file'] not in excluded)),
               excluded_boundary_files=sorted(excluded),files=rows,errors=errors,
               obstacle_csv_limitation='CSV contains heights/areas but no geometry; not used as segmentation ground truth.')
    (report_dir/'geneva_roof_audit.json').write_text(json.dumps(audit,indent=2))
    sheet=Image.new('RGB',(1280,960))
    for i,sample in enumerate(samples):sheet.paste(sample,((i%4)*320,(i//4)*320))
    sheet.save(report_dir/'geneva_roof_overlays.jpg')
    print(json.dumps({k:v for k,v in audit.items() if k not in {'files'}},indent=2))

if __name__=='__main__':main()
