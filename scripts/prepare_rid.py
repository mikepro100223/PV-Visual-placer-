"""Convert reviewed RID masks with geographic overlap removed between splits."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from pyproj import Transformer
from shapely.geometry import box
from shapely.strtree import STRtree
import yaml

from prepare_swiss import mask_polygons

ROOT=Path(__file__).resolve().parents[1]
# Verified against TUMFTM/RID definitions.py and mask_generation.py.
# Background is 8, not 0. Preserve every foreground class separately.
NAMES={0:'pv_installation',1:'dormer',2:'skylight',3:'ladder',4:'chimney',5:'shadow',6:'tree',7:'unknown_obstacle'}
COLORS=np.array([[60,160,255],[255,80,50],[255,190,30],[240,90,210],[180,75,255],[90,90,90],[60,220,80],[255,245,120]])

def main():
    source=ROOT/'data/raw/rid'; target=ROOT/'data/processed/rid'; report_dir=ROOT/'data/reports'
    target.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    projected=Transformer.from_crs(4326,25832,always_xy=True)
    records=[]
    for split in ['test','val','train']:
        names=(source/'filenames_train_val_test_split'/f'{split}_filenames_1_rev.txt').read_text().splitlines()
        for name in names:
            path=source/'images_roof_centered_geotiff'/(Path(name).stem+'.tif')
            with Image.open(path) as image:
                x,y=image.tag_v2[33922][3:5];dx,dy=image.tag_v2[33550][:2]
                w,h=image.size
                corners=[projected.transform(a,b) for a,b in [(x,y),(x+w*dx,y),(x+w*dx,y-h*dy),(x,y-h*dy)]]
                xs,ys=zip(*corners)
                bounds=(min(xs),min(ys),max(xs),max(ys))
                records.append(dict(file=name,split=split,bounds=bounds,gsd_m=(max(xs)-min(xs))/w))
    if len(records)!=1880 or len({r['file'] for r in records})!=1880:
        raise ValueError('RID source split must cover 1880 distinct image/mask pairs')
    kept=[];excluded=[];higher=[]
    for split in ['test','val','train']:
        tree=STRtree(higher) if higher else None
        selected=[]
        for row in [r for r in records if r['split']==split]:
            geometry=box(*row['bounds'])
            if tree is not None and len(tree.query(geometry.buffer(2),predicate='intersects')):
                excluded.append(row)
            else:
                kept.append(row);selected.append(geometry)
        higher.extend(selected)
    counts=Counter();instances={s:Counter() for s in ['train','val','test']};ious=[];samples=[];seen={};accepted=[];duplicates=[]
    for row in kept:
        name,split=row['file'],row['split']
        with Image.open(source/'images_roof_centered_geotiff'/(Path(name).stem+'.tif')) as original:
            image=original.convert('RGB')
        mask=np.array(Image.open(source/'masks_superstructures_reviewed'/name))
        if mask.shape!=(512,512) or not set(np.unique(mask)).issubset(set(range(9))):
            raise ValueError('Unexpected RID mask format: '+name)
        digest=hashlib.sha256(np.array(image).tobytes()).hexdigest()
        if digest in seen:
            duplicates.append(dict(file=name,duplicate_of=seen[digest],split=split))
            continue
        seen[digest]=name
        accepted.append(row)
        lines=[]
        for value,label in NAMES.items():
            polygons,iou=mask_polygons(mask,value,value,min_area=3)
            lines.extend(polygons);instances[split][label]+=len(polygons)
            if np.any(mask==value): ious.append(iou)
        row.update(instances=len(lines),image_sha256=digest)
        counts[split]+=1
        for kind in ['images','labels']: (target/kind/split).mkdir(parents=True,exist_ok=True)
        image.save(target/'images'/split/name)
        (target/'labels'/split/(Path(name).stem+'.txt')).write_text('\n'.join(lines),encoding='utf-8')
        if len(samples)<12 and np.any(mask==0) and np.any(mask==4):
            rgb=np.array(image)
            for cls,color in enumerate(COLORS): rgb[mask==cls]=(rgb[mask==cls]*.45+color*.55).astype(np.uint8)
            sample=Image.fromarray(rgb).resize((320,320))
            ImageDraw.Draw(sample).text((5,5),name+' '+split,fill='white',stroke_width=1,stroke_fill='black')
            samples.append(sample)
    kept=accepted
    for split in ['train','val','test']:
        (target/(split+'.txt')).write_text('\n'.join((target/'images'/split/r['file']).as_posix() for r in kept if r['split']==split),encoding='utf-8')
    config=dict(path=target.as_posix(),train='train.txt',val='val.txt',test='test.txt',names=NAMES)
    (target/'dataset.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    report=dict(source='TUM RID reviewed masks, CC-BY-NC 4.0',source_count=len(records),counts=dict(counts),
                class_instances={s:dict(c) for s,c in instances.items()},
                exclusions_due_to_overlap=len(excluded),excluded_files=excluded,duplicates_removed=duplicates,
                mean_conversion_iou=float(np.mean(ious)),minimum_conversion_iou=min(ious),
                mean_gsd_m=float(np.mean([r['gsd_m'] for r in kept])),files=kept)
    (report_dir/'rid_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1280,960))
    for i,sample in enumerate(samples):sheet.paste(sample,((i%4)*320,(i//4)*320))
    sheet.save(report_dir/'rid_overlays.jpg')
    print(json.dumps({k:v for k,v in report.items() if k not in {'files','excluded_files'}},indent=2))

if __name__=='__main__':main()
