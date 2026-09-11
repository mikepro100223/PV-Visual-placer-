"""Use only locally trained checkpoints; never relabel a generic model as solar."""
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from app.detections import ALIASES,KINDS,normalise,merge_detections,pixel_to_world

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/ultralytics'))
LOCK = threading.Lock()
LOADED = {}
STATUS_CACHE = {}

def checkpoint_path(name):
    if name=='rid' and os.environ.get('OBSTACLE_MODEL_PATH'):
        path=Path(os.environ['OBSTACLE_MODEL_PATH'])
        return path if path.is_absolute() else ROOT/path
    return ROOT/'models'/f'{name}_best.pt'

def status():
    result = {}
    for name in ['swiss','rid','roof']:
        report = ROOT/'models'/f'{name}_status.json'
        try:
            if report.exists():STATUS_CACHE[name]=json.loads(report.read_text())
        except (OSError,json.JSONDecodeError):
            pass  # Keep the last complete status during a Windows file lock.
        result[name] = dict(STATUS_CACHE.get(name,{'state':'not_trained'}))
        if name=='rid' and os.environ.get('OBSTACLE_MODEL_PATH'):
            result[name]={'state':'ready' if checkpoint_path(name).is_file() else 'not_trained','external_checkpoint':True}
        result[name]['available'] = checkpoint_path(name).is_file()
        result[name]['checkpoint']=checkpoint_path(name).name
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
            path = checkpoint_path(dataset)
            if not path.exists():
                continue
            stamp = (str(path.resolve()),path.stat().st_mtime_ns)
            if dataset not in LOADED or LOADED[dataset][0] != stamp:
                LOADED[dataset] = (stamp,YOLO(path))
            model = LOADED[dataset][1]
            if dataset=='rid':
                names={ALIASES.get(str(n),str(n)) for n in model.names.values()}
                if model.task!='segment' or not names.issubset(KINDS-{'roof'}) or not names-{'pv_installation'}:
                    raise ValueError('Obstacle checkpoint must segment rooftop obstacles; PV-only or generic COCO weights are unsupported.')
                # Abbas contributes a full-image obstacle pass only. Native
                # crops below retain the original Yucan PV predictions exactly.
                if max(image.size)>768:
                    result=model.predict(image,conf=obstacle_confidence,imgsz=1024,
                                         device=device,retina_masks=True,verbose=False)[0]
                    if result.masks is not None:
                        for coords,cls,score in zip(result.masks.xy,result.boxes.cls.tolist(),result.boxes.conf.tolist()):
                            kind=ALIASES.get(model.names[int(cls)],model.names[int(cls)])
                            if kind=='pv_installation' or len(coords)<3:continue
                            geom=pixel_to_world(Polygon(coords).buffer(0),bounds,image.size)
                            predictions.append(dict(geometry=geom,label=kind,confidence=round(score,3),model='rid'))
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
                        label=ALIASES.get(model.names[int(cls)],model.names[int(cls)])
                        # The Swiss checkpoint is the high-quality, in-domain
                        # PV specialist. RID remains a fallback for PV only
                        # when that checkpoint is unavailable.
                        if (dataset=='rid' and label=='pv_installation'
                                and checkpoint_path('swiss').is_file()):
                            continue
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
    obstacles=merge_detections([d for d in predictions if d['model']=='rid'])
    return normalise([d for d in predictions if d['model']!='rid'])+obstacles
