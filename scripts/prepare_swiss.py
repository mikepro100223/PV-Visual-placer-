"""Audit Swiss PV masks and create geographically grouped YOLO11 data."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]

def mask_polygons(mask, value, class_id, min_area=3):
    binary = (mask == value).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    lines = []
    reconstructed = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        cv2.fillPoly(reconstructed, [points], 1)
        coords = points.astype(float) / [w, h]
        lines.append(str(class_id) + ' ' + ' '.join(f'{v:.7f}' for v in coords.flat))
    union = np.count_nonzero(binary | reconstructed)
    iou = np.count_nonzero(binary & reconstructed) / union if union else 1.0
    return lines, iou

def split_group(key):
    bucket = int(hashlib.sha256(('42:' + key).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 70 else 'val' if bucket < 85 else 'test'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/raw/swiss')
    parser.add_argument('--task',choices=['swiss','roof'],default='swiss')
    args=parser.parse_args()
    source = args.source.resolve()
    target = ROOT / 'data/processed' / args.task
    report_dir = ROOT / 'data/reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    counts, groups, values = Counter(), {}, Counter()
    rows, samples, seen = [], [], {}
    for path in sorted((source / 'images').glob('*.jpg')):
        mask_path = source / ('labels' if args.task=='swiss' else 'roofs/masks') / (path.stem + '.png')
        if not mask_path.is_file():
            raise ValueError(f'Missing mask: {path.name}')
        image = Image.open(path).convert('RGB')
        mask = np.array(Image.open(mask_path))
        if mask.ndim != 2 or image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f'Invalid mask shape for {path.name}')
        vals = np.unique(mask).tolist()
        if not set(vals).issubset({0, 1, 255}):
            raise ValueError(f'Unexpected mask values: {vals}')
        values.update(vals)
        match = re.search(r'_(\d{4})_(\d+\.\d+)-(\d+\.\d+)', path.stem)
        if not match:
            raise ValueError(f'Cannot locate image: {path.name}')
        year, easting, northing = match.groups()
        # Names encode the bottom-left corner of 100 m Swiss LV95 chips in km.
        x, y = round(float(easting) * 1000), round(float(northing) * 1000)
        group = f'{x // 2000}:{y // 2000}'
        split = split_group(group)
        groups[group] = split
        digest = hashlib.sha256(np.array(image).tobytes()).hexdigest()
        if digest in seen:
            counts['duplicates_removed'] += 1
            continue
        seen[digest] = path.name
        mask = (mask > 0).astype(np.uint8)
        labels, iou = mask_polygons(mask, 1, 0)
        record = dict(file=path.name, split=split, group=group, year=int(year), x=x, y=y,
                      instances=len(labels), foreground_pixels=int(mask.sum()), conversion_iou=iou,
                      image_sha256=digest)
        rows.append(record)
        for kind in ['images', 'labels']:
            (target / kind / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / 'images' / split / path.name)
        (target / 'labels' / split / (path.stem + '.txt')).write_text('\n'.join(labels), encoding='utf-8')
        counts[split] += 1
        counts[split + '_positive' if labels else split + '_negative'] += 1
        if labels and len(samples) < 12:
            pixels = np.array(image)
            pixels[mask > 0] = (pixels[mask > 0] * 0.45 + np.array([255, 80, 30]) * 0.55).astype(np.uint8)
            sample = Image.fromarray(pixels).resize((320, 320))
            ImageDraw.Draw(sample).text((5, 5), path.stem[-25:], fill='white', stroke_width=1, stroke_fill='black')
            samples.append(sample)
    # Chips are 100 m wide: remove training chips immediately adjacent to a
    # held-out chip at a block boundary. Repeat acquisitions share their block.
    held = [(r['x'], r['y']) for r in rows if r['split'] != 'train']
    train_rows = [r for r in rows if r['split'] == 'train']
    excluded = {r['file'] for r in train_rows if any(abs(r['x']-x) <= 100 and abs(r['y']-y) <= 100 for x,y in held)}
    # Use explicit manifests so reruns cannot accidentally include stale files.
    for split in ['train', 'val', 'test']:
        included = [r for r in rows if r['split'] == split and r['file'] not in excluded]
        (target / (split + '.txt')).write_text('\n'.join((target/'images'/split/r['file']).as_posix() for r in included), encoding='utf-8')
    config = dict(path=target.as_posix(), train='train.txt', val='val.txt', test='test.txt', names={0:'pv_installation' if args.task=='swiss' else 'roof'})
    (target / 'dataset.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    report = dict(source=str(source),task=args.task, counts=dict(counts),
                  excluded_train_boundary_files=sorted(excluded), mask_values=dict(values), groups=groups,
                  minimum_conversion_iou=min(r['conversion_iou'] for r in rows),
                  mean_conversion_iou=float(np.mean([r['conversion_iou'] for r in rows])), files=rows)
    (report_dir/f'{args.task}_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sheet = Image.new('RGB', (1280, 960))
    for i, sample in enumerate(samples):
        sheet.paste(sample, ((i%4)*320, (i//4)*320))
    sheet.save(report_dir/f'{args.task}_overlays.jpg')
    print(json.dumps({k:v for k,v in report.items() if k not in {'files','groups'}}, indent=2))

if __name__ == '__main__':
    main()
