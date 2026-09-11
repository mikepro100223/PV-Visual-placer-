"""Evaluate end-to-end detection consistency on reviewed Swiss roof cases.

The cases reproduce the roofs supplied during the September 2026 error
analysis. They are regression cases, not a substitute for held-out labelled
mask metrics.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union


SITES={
    'narrow': (47.395899926402244,8.035635352134706),
    'flat': (47.3960170366071,8.035194128751757),
    'complex_a': (47.39171647024269,8.0453959107399),
    'complex_b': (47.39061789480003,8.04778039455414),
    'complex_c': (47.39017573441912,8.047635555267336),
    'simple_flat': (47.39083942755903,8.053417056798937),
}

# Captured with the published Yucan obstacle checkpoint after assigning PV
# ownership to the substantially stronger in-domain Swiss checkpoint. These
# are system-consistency values, not ground truth.
BASELINE={
    'complex_c': dict(panel_count=411,pv_area_m2=462.35,pv_hard_overlap_m2=19.82),
}


def _union(geometries):
    return unary_union(geometries) if geometries else Polygon()


def summarise(name,data):
    detections=[]
    by_source=defaultdict(lambda:dict(count=0,area_m2=0.0))
    for item in data.get('detections',[]):
        geometry=Polygon(item['polygon'],item.get('holes',[]))
        if geometry.is_empty:continue
        detections.append((item,geometry))
        key=f"{item['kind']}:{item.get('source','unknown')}"
        by_source[key]['count']+=1
        by_source[key]['area_m2']+=geometry.area
    pv=_union([g for item,g in detections if item['kind']=='pv_installation'])
    hard=_union([g for item,g in detections
                 if item['kind']!='pv_installation' and item.get('blocks_placement',True)])
    advisory=_union([g for item,g in detections if not item.get('blocks_placement',True)])
    overlap=pv.intersection(hard).area
    return dict(
        site=name,
        building_id=data.get('building_id'),
        panel_count=data.get('panel_count',0),
        usable_area_m2=data.get('usable_area_m2',0),
        pv_area_m2=round(pv.area,2),
        hard_obstacle_area_m2=round(hard.area,2),
        advisory_area_m2=round(advisory.area,2),
        pv_hard_overlap_m2=round(overlap,2),
        pv_hard_overlap_ratio=round(overlap/pv.area,4) if pv.area else 0,
        detections={key:{'count':value['count'],'area_m2':round(value['area_m2'],2)}
                    for key,value in sorted(by_source.items())},
    )


def gates(cases):
    complex_case=next(case for case in cases if case['site']=='complex_c')
    baseline=BASELINE['complex_c']
    checks={
        'preserve_existing_pv_area': abs(complex_case['pv_area_m2']-baseline['pv_area_m2'])<=1,
        'reduce_pv_obstacle_conflict_by_65_percent':
            complex_case['pv_hard_overlap_m2']<=baseline['pv_hard_overlap_m2']*.35,
        'avoid_large_layout_jump':
            abs(complex_case['panel_count']-baseline['panel_count'])<=5,
        'shadows_are_advisory': all(
            not item.get('blocks_placement',True)
            for case in cases for item in case.get('_raw_detections',[])
            if item['kind']=='shadow'),
    }
    return dict(passed=all(checks.values()),checks=checks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--api',default='http://127.0.0.1:8000')
    parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--obstacle-confidence',type=float,default=.30)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cases=[]
    for name,(lat,lon) in SITES.items():
        response=requests.get(
            f'{args.api}/api/analyze',
            params=dict(lat=lat,lon=lon,confidence=args.confidence,
                        obstacle_confidence=args.obstacle_confidence),
            timeout=240,
        )
        response.raise_for_status()
        raw=response.json()
        case=summarise(name,raw)
        case['_raw_detections']=raw.get('detections',[])
        cases.append(case)
    quality=gates(cases)
    for case in cases:case.pop('_raw_detections',None)
    report=dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        api=args.api,
        parameters=dict(confidence=args.confidence,
                        obstacle_confidence=args.obstacle_confidence),
        baseline_checkpoint_sha256='c9be26fb2f7d64eca37b600e5237232d581c6d421b92a59bf2f1244606e3e044',
        baseline=BASELINE,
        cases=cases,
        quality_gates=quality,
        interpretation='Regression benchmark only; labelled held-out datasets remain authoritative for model promotion.',
    )
    rendered=json.dumps(report,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+'\n',encoding='utf-8')
    print(rendered)
    raise SystemExit(0 if quality['passed'] else 1)


if __name__=='__main__':main()
