from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from app import inference


def test_original_yucan_pv_sources_and_independent_obstacle_confidence(tmp_path,monkeypatch):
    import ultralytics
    import torch
    (tmp_path/'models').mkdir()
    for name in ['swiss','rid','roof']:(tmp_path/'models'/f'{name}_best.pt').touch()
    monkeypatch.setattr(inference,'ROOT',tmp_path)
    monkeypatch.setattr(inference,'LOADED',{})
    monkeypatch.delenv('OBSTACLE_MODEL_PATH',raising=False)
    monkeypatch.delenv('PV_INFERENCE_DEVICE',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    calls=[]
    class Model:
        task='segment'
        def __init__(self,path):
            self.dataset=path.stem.split('_')[0]
            self.names={'swiss':{0:'pv_installation'},'rid':{0:'pv_installation',1:'chimney'},'roof':{0:'roof'}}[self.dataset]
        def predict(self,image,**kwargs):
            calls.append((self.dataset,kwargs['conf']))
            polygon=np.array([[10,10],[30,10],[30,30],[10,30]])
            n=len(self.names)
            return [SimpleNamespace(masks=SimpleNamespace(xy=[polygon]*n),
                    boxes=SimpleNamespace(cls=np.arange(n),conf=np.full(n,.9)))]
    monkeypatch.setattr(ultralytics,'YOLO',Model)
    detected=inference.predict(Image.new('RGB',(400,400)),(0,0,40,40),.25,.6)
    assert [(d['label'],d['model']) for d in detected if d['label']=='pv_installation']==[('pv_installation','swiss'),('pv_installation','rid')]
    assert any(d['label']=='chimney' for d in detected)
    assert ('swiss',.25) in calls and ('rid',.25) in calls


def test_external_obstacle_checkpoint_replaces_rid_instead_of_loading_both(tmp_path,monkeypatch):
    monkeypatch.setattr(inference,'ROOT',tmp_path)
    monkeypatch.setenv('OBSTACLE_MODEL_PATH','models/obstacle_best.pt')
    assert inference.checkpoint_path('rid')==tmp_path/'models/obstacle_best.pt'
    assert inference.checkpoint_path('swiss')==tmp_path/'models/swiss_best.pt'
