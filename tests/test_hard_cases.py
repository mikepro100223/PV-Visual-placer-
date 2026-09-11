from scripts.evaluate_hard_cases import gates,summarise


def detection(kind,polygon,source='rid',blocks=True):
    return dict(kind=kind,polygon=polygon,holes=[],source=source,
                blocks_placement=blocks)


def test_summary_measures_cross_class_overlap_and_advisory_area():
    data={
        'building_id': 7,
        'panel_count': 12,
        'usable_area_m2': 30,
        'detections': [
            detection('pv_installation',[(0,0),(10,0),(10,10),(0,10)],'swiss'),
            detection('chimney',[(8,0),(12,0),(12,10),(8,10)]),
            detection('shadow',[(0,0),(2,0),(2,2),(0,2)],blocks=False),
        ],
    }
    report=summarise('sample',data)
    assert report['pv_area_m2']==100
    assert report['hard_obstacle_area_m2']==40
    assert report['pv_hard_overlap_m2']==20
    assert report['pv_hard_overlap_ratio']==.2
    assert report['advisory_area_m2']==4
    assert report['detections']['chimney:rid']=={'count':1,'area_m2':40}


def test_summary_handles_no_pv_without_division_by_zero():
    report=summarise('empty',{'detections':[],'panel_count':0,'usable_area_m2':0})
    assert report['pv_area_m2']==0
    assert report['pv_hard_overlap_ratio']==0


def test_candidate_gate_uses_specialist_swiss_pv_baseline():
    candidate=dict(site='complex_c',panel_count=409,pv_area_m2=462.35,
                   pv_hard_overlap_m2=3.3,_raw_detections=[])
    result=gates([candidate])
    assert result['passed'] is True
