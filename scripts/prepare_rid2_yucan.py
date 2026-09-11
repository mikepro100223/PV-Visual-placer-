"""Map the audited RID2 split onto Yucan's eight-class YOLO head.

This creates a separate derived dataset and never modifies RID2 source labels.
Shadow and tree stay in the head for checkpoint compatibility but receive no
invented supervision because RID2 does not annotate those classes.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil


ROOT=Path(__file__).resolve().parents[1]
CLASS_MAP={
    0: 0,   # solar_panel -> pv_installation
    1: 4,   # chimney
    2: 2,   # skylight
    3: 1,   # dormer
    4: 2,   # roof_window -> skylight
    5: 7,   # hvac -> unknown_obstacle
    6: 7,   # tv_dish -> unknown_obstacle
    7: 3,   # ladder
    8: 7,   # balcony -> unknown_obstacle
    9: 7,   # wall -> unknown_obstacle
    10: 7,  # other -> unknown_obstacle
}
NAMES=['pv_installation','dormer','skylight','ladder','chimney','shadow','tree','unknown_obstacle']


def convert_label_line(line):
    fields=line.strip().split()
    if not fields:return ''
    try:
        source_class=int(fields[0])
        coordinates=[float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(f'Invalid YOLO segmentation label: {line!r}') from exc
    if source_class not in CLASS_MAP:
        raise ValueError(f'Unsupported RID2 class {source_class}')
    if len(coordinates)<6 or len(coordinates)%2:
        raise ValueError('A segmentation polygon needs at least three x/y points')
    if any(not math.isfinite(value) or not 0<=value<=1 for value in coordinates):
        raise ValueError('Segmentation coordinates must be finite and normalized')
    return f"{CLASS_MAP[source_class]} {' '.join(fields[1:])}\n"


def convert_label_lines(lines):
    converted=[]
    seen=set()
    for line in lines:
        mapped=convert_label_line(line)
        if mapped and mapped not in seen:
            converted.append(mapped)
            seen.add(mapped)
    return converted


def _link_or_copy(source,target):
    if target.exists():return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def clear_ultralytics_caches(target):
    """Remove only regenerable label caches inside the derived dataset."""
    removed=[]
    for cache in sorted((target/'labels').glob('*.cache')):
        if cache.is_file():
            cache.unlink()
            removed.append(cache)
    return removed


def prepare(source,target):
    if source.resolve()==target.resolve():
        raise ValueError('Derived dataset must not overwrite the RID2 source')
    clear_ultralytics_caches(target)
    source_manifest=json.loads((source/'splits.json').read_text(encoding='utf-8'))
    counts={}; class_counts=Counter()
    for split in ('train','val','test'):
        source_images=source/'images'/split
        source_labels=source/'labels'/split
        target_images=target/'images'/split
        target_labels=target/'labels'/split
        target_images.mkdir(parents=True,exist_ok=True)
        target_labels.mkdir(parents=True,exist_ok=True)
        images=sorted(path for path in source_images.iterdir() if path.is_file())
        for image in images:
            label=source_labels/f'{image.stem}.txt'
            if not label.is_file():raise FileNotFoundError(f'Missing label for {image}')
            _link_or_copy(image,target_images/image.name)
            converted=convert_label_lines(
                label.read_text(encoding='utf-8').splitlines(keepends=True))
            for mapped in converted:
                class_counts[int(mapped.split(maxsplit=1)[0])]+=1
            (target_labels/label.name).write_text(''.join(converted),encoding='utf-8')
        counts[split]=len(images)
    yaml=['path: '+str(target.resolve()),'train: images/train','val: images/val','test: images/test','names:']
    yaml.extend(f'  {index}: {name}' for index,name in enumerate(NAMES))
    (target/'dataset.yaml').write_text('\n'.join(yaml)+'\n',encoding='utf-8')
    manifest=dict(
        derived_from=source_manifest.get('source',{}),
        source_manifest=str((source/'splits.json').resolve()),
        source_splits=source_manifest.get('splits'),
        image_counts=counts,
        class_map={str(key):value for key,value in CLASS_MAP.items()},
        names=NAMES,
        polygon_counts={NAMES[key]:class_counts[key] for key in range(len(NAMES))},
        limitations=[
            'RID2 has no shadow or tree labels; no labels were invented for classes 5 and 6.',
            'The test split is evaluation-only and must not be used for training or threshold selection.',
            'This mapping is a derived compatibility dataset; original RID2 files remain unchanged.',
        ],
    )
    (target/'splits.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/processed/rid2')
    parser.add_argument('--target',type=Path,default=ROOT/'data/processed/rid2-yucan')
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.target),indent=2,sort_keys=True))


if __name__=='__main__':main()
