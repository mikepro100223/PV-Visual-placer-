from scripts.prepare_rid2_yucan import CLASS_MAP,clear_ultralytics_caches,convert_label_line,convert_label_lines


def test_rid2_classes_map_to_yucan_head_without_index_drift():
    assert CLASS_MAP=={0:0,1:4,2:2,3:1,4:2,5:7,6:7,7:3,8:7,9:7,10:7}


def test_label_conversion_preserves_segmentation_coordinates():
    source='4 0.1 0.2 0.3 0.4 0.5 0.6\n'
    assert convert_label_line(source)=='2 0.1 0.2 0.3 0.4 0.5 0.6\n'


def test_label_conversion_rejects_invalid_polygon_or_unknown_class():
    for source in ('99 0.1 0.2 0.3 0.4 0.5 0.6\n','1 0.1 0.2 0.3 0.4\n'):
        try:convert_label_line(source)
        except ValueError:pass
        else:raise AssertionError(f'accepted invalid label: {source!r}')


def test_mapping_deduplicates_polygons_that_collapse_into_unknown_class():
    polygon='0.1 0.2 0.3 0.4 0.5 0.6\n'
    assert convert_label_lines(['5 '+polygon,'10 '+polygon])==['7 '+polygon]


def test_clear_ultralytics_caches_only_removes_derived_label_caches(tmp_path):
    labels=tmp_path/'labels'
    labels.mkdir()
    cache=labels/'train.cache'
    unrelated=tmp_path/'keep.cache'
    cache.write_text('stale',encoding='utf-8')
    unrelated.write_text('keep',encoding='utf-8')

    assert clear_ultralytics_caches(tmp_path)==[cache]
    assert not cache.exists()
    assert unrelated.read_text(encoding='utf-8')=='keep'
