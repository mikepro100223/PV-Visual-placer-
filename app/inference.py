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
STATUS_CACHE = {}

def status():
    result = {}
    for name in ['swiss','rid','roof']:
        report = ROOT/'models'/f'{name}_status.json'
        try:
            if report.exists():STATUS_CACHE[name]=json.loads(report.read_text())
        except (OSError,json.JSONDecodeError):
            pass  # Keep the last complete status during a Windows file lock.
        result[name] = dict(STATUS_CACHE.get(name,{'state':'not_trained'}))
        result[name]['available'] = (ROOT/'models'/f'{name}_best.pt').exists()
    return result

def predict(image, bounds, confidence, obstacle_confidence=None):
    from ultralytics import YOLO
    minx,miny,maxx,maxy = bounds
    width,height = image.size
    predictions = []
    obstacle_confidence=confidence if obstacle_confidence is None else obstacle_confidence
    import torch
    # Honor the requested 4090 for inference too; configurable CPU fallback is
    # available for machines without CUDA, not used on this training machine.
    device = os.environ.get('PV_INFERENCE_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
    with LOCK:
        for dataset in ['swiss','rid','roof']:
            path = ROOT/'models'/f'{dataset}_best.pt'
            if not path.exists():
                continue
            stamp = path.stat().st_mtime_ns
            if dataset not in LOADED or LOADED[dataset][0] != stamp:
                LOADED[dataset] = (stamp,YOLO(path))
            model = LOADED[dataset][1]
            # Match training object scale: Swiss chips cover 100 m; RID crops
            # are smaller. Overlapping crops retain tiny objects on wide roofs.
            tile = 512 if dataset == 'rid' else 1000
            stride = int(tile*.75)
            xs = sorted(set([*range(0,max(width-tile,0)+1,stride),max(width-tile,0)]))
            ys = sorted(set([*range(0,max(height-tile,0)+1,stride),max(height-tile,0)]))
            for x0 in xs:
                for y0 in ys:
                    raw_crop = image.crop((x0,y0,min(width,x0+tile),min(height,y0+tile)))
                    crop = Image.new('RGB',(tile,tile),(114,114,114))
                    crop.paste(raw_crop,(0,0))
                    result = model.predict(crop,conf=min(confidence,obstacle_confidence) if dataset=='rid' else confidence,imgsz=1024,
                                           device=device,retina_masks=True,verbose=False)[0]
                    if result.masks is None:
                        continue
                    for coords, cls, score in zip(result.masks.xy,result.boxes.cls.tolist(),result.boxes.conf.tolist()):
                        label=model.names[int(cls)]
                        threshold=obstacle_confidence if dataset=='rid' and label!='pv_installation' else confidence
                        if score<threshold:
                            continue
                        if len(coords)<3:
                            continue
                        points = [(minx+(float(x)+x0)/width*(maxx-minx),
                                   maxy-(float(y)+y0)/height*(maxy-miny)) for x,y in coords]
                        from shapely.geometry import box
                        geom = Polygon(points).buffer(0).intersection(box(minx,miny,maxx,maxy))
                        if geom.is_empty:
                            continue
                        predictions.append(dict(geometry=geom,label=label,
                                                confidence=round(score,3),model=dataset))
    return predictions
