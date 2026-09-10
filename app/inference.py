"""Use only locally trained checkpoints; never relabel a generic model as solar."""
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/ultralytics'))
LOCK = threading.Lock()
LOADED = {}

def status():
    result = {}
    for name in ['swiss','rid']:
        report = ROOT/'models'/f'{name}_status.json'
        result[name] = json.loads(report.read_text()) if report.exists() else {'state':'not_trained'}
        result[name]['available'] = (ROOT/'models'/f'{name}_best.pt').exists()
    return result

def predict(image, bounds, confidence):
    from ultralytics import YOLO
    minx,miny,maxx,maxy = bounds
    width,height = image.size
    predictions = []
    import torch
    # Honor the requested 4090 for inference too; configurable CPU fallback is
    # available for machines without CUDA, not used on this training machine.
    device = os.environ.get('PV_INFERENCE_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
    with LOCK:
        for dataset in ['swiss','rid']:
            path = ROOT/'models'/f'{dataset}_best.pt'
            if not path.exists():
                continue
            stamp = path.stat().st_mtime_ns
            if dataset not in LOADED or LOADED[dataset][0] != stamp:
                LOADED[dataset] = (stamp,YOLO(path))
            model = LOADED[dataset][1]
            # Match training object scale: Swiss chips cover 100 m; RID crops
            # are smaller. Overlapping crops retain tiny objects on wide roofs.
            tile = 1000 if dataset == 'swiss' else 512
            stride = int(tile*.75)
            xs = sorted(set([*range(0,max(width-tile,0)+1,stride),max(width-tile,0)]))
            ys = sorted(set([*range(0,max(height-tile,0)+1,stride),max(height-tile,0)]))
            for x0 in xs:
                for y0 in ys:
                    raw_crop = image.crop((x0,y0,min(width,x0+tile),min(height,y0+tile)))
                    crop = Image.new('RGB',(tile,tile),(114,114,114))
                    crop.paste(raw_crop,(0,0))
                    result = model.predict(crop,conf=confidence,imgsz=1024,
                                           device=device,retina_masks=True,verbose=False)[0]
                    if result.masks is None:
                        continue
                    for coords, cls, score in zip(result.masks.xy,result.boxes.cls.tolist(),result.boxes.conf.tolist()):
                        if len(coords)<3:
                            continue
                        points = [(minx+(float(x)+x0)/width*(maxx-minx),
                                   maxy-(float(y)+y0)/height*(maxy-miny)) for x,y in coords]
                        from shapely.geometry import box
                        geom = Polygon(points).buffer(0).intersection(box(minx,miny,maxx,maxy))
                        if geom.is_empty:
                            continue
                        predictions.append(dict(geometry=geom,label=model.names[int(cls)],
                                                confidence=round(score,3),model=dataset))
    return predictions
