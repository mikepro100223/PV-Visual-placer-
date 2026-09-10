"""Compare obstacle inference on validation masks, without touching the test set."""
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/ultralytics'))

def main():
    from ultralytics import YOLO
    folder=ROOT/'data/reports/obstacle_inference'
    folder.mkdir(exist_ok=True)
    snapshot=folder/'candidate.pt'
    shutil.copy2(ROOT/'runs/segment/rid3/weights/best.pt',snapshot)
    files=(ROOT/'data/processed/rid/val.txt').read_text().splitlines()[::7]
    reports=[]
    for name,checkpoint,tiles in [('preview',ROOT/'models/rid_best.pt',[512]),
                                  ('candidate',snapshot,[512]),
                                  ('candidate_zoom',snapshot,[512,256])]:
        model=YOLO(checkpoint)
        counts={threshold:np.zeros(3,dtype=np.int64) for threshold in [.15,.25,.35]}
        for filename in files:
            path=Path(filename)
            image=Image.open(path).convert('RGB')
            mask=np.array(Image.open(ROOT/'data/raw/rid/masks_superstructures_reviewed'/f'{path.stem}.png'))
            truth=np.isin(mask,[1,2,3,4,7])
            predicted={threshold:np.zeros(mask.shape,dtype=np.uint8) for threshold in counts}
            for tile in tiles:
                stride=tile*3//4
                for x in sorted(set([*range(0,513-tile,stride),512-tile])):
                    for y in sorted(set([*range(0,513-tile,stride),512-tile])):
                        result=model.predict(image.crop((x,y,x+tile,y+tile)),imgsz=1024,conf=.15,
                                             device=0,retina_masks=True,verbose=False)[0]
                        if result.masks is None:continue
                        for polygon,cls,conf in zip(result.masks.xy,result.boxes.cls.tolist(),result.boxes.conf.tolist()):
                            if int(cls) not in [1,2,3,4,7]:continue
                            polygon=np.round(polygon+np.array([x,y])).astype(np.int32)
                            for threshold in counts:
                                if conf>=threshold:cv2.fillPoly(predicted[threshold],[polygon],1)
            for threshold,p in predicted.items():
                p=p>0
                counts[threshold]+=np.array([np.count_nonzero(truth&p),np.count_nonzero(~truth&p),np.count_nonzero(truth&~p)])
        for threshold,(tp,fp,fn) in counts.items():
            row=dict(name=name,confidence=threshold,images=len(files),iou=float(tp/(tp+fp+fn)),
                     precision=float(tp/(tp+fp)),recall=float(tp/(tp+fn)))
            reports.append(row);print(json.dumps(row),flush=True)
        del model
    (folder/'validation.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':main()
