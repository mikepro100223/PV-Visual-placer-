"""Measure pixel coverage against original held-out masks at the UI threshold."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/ultralytics'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['swiss','rid'],required=True)
    parser.add_argument('--confidence',type=float,default=.25)
    args=parser.parse_args()
    from ultralytics import YOLO
    model=YOLO(ROOT/'models'/f'{args.dataset}_best.pt')
    manifest=ROOT/'data/processed'/args.dataset/'test.txt'
    totals=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    images=[]
    for filename in manifest.read_text().splitlines():
        path=Path(filename)
        if args.dataset=='swiss':
            source=ROOT/'datasets/kaggle-swiss-solar-panels-segmentation/labels'/f'{path.stem}.png'
            if not source.exists():source=ROOT/'data/raw/swiss/labels'/f'{path.stem}.png'
            mask=(np.array(Image.open(source))>0).astype(np.uint8)
        else:
            source=ROOT/'data/raw/rid/masks_superstructures_reviewed'/f'{path.stem}.png'
            mask=np.array(Image.open(source))
        result=model.predict(path,imgsz=1024,conf=args.confidence,device=0,retina_masks=True,verbose=False)[0]
        predicted={int(k):np.zeros(mask.shape,dtype=np.uint8) for k in model.names}
        if result.masks is not None:
            for polygon,cls in zip(result.masks.xy,result.boxes.cls.tolist()):
                if len(polygon)>=3:cv2.fillPoly(predicted[int(cls)],[np.round(polygon).astype(np.int32)],1)
        row=dict(file=path.name,classes={})
        for cls,name in model.names.items():
            truth=mask==1 if args.dataset=='swiss' else mask==int(cls)
            pred=predicted[int(cls)]>0
            tp=int(np.count_nonzero(truth&pred));fp=int(np.count_nonzero(~truth&pred));fn=int(np.count_nonzero(truth&~pred))
            totals[name]+=np.array([tp,fp,fn])
            row['classes'][name]=dict(true_positive_pixels=tp,false_positive_pixels=fp,false_negative_pixels=fn,
                                      iou=tp/(tp+fp+fn) if tp+fp+fn else None)
        images.append(row)
    metrics={}
    for name,(tp,fp,fn) in totals.items():
        tp,fp,fn=int(tp),int(fp),int(fn)
        metrics[name]=dict(iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                           dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                           pixel_precision=tp/(tp+fp) if tp+fp else None,
                           pixel_recall=tp/(tp+fn) if tp+fn else None)
    report=dict(dataset=args.dataset,split='test',confidence=args.confidence,images=len(images),
                ground_truth='original raster masks, not converted polygon labels',classes=metrics,per_image=images)
    (ROOT/'models'/f'{args.dataset}_coverage.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'},indent=2))

if __name__=='__main__':main()
