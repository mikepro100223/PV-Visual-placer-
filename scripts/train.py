"""Train a real YOLO11 segmentation checkpoint and publish it locally."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'data/ultralytics'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['swiss', 'rid'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--wait-for', choices=['swiss','rid'], default=None)
    parser.add_argument('--mask-ratio', type=int, choices=[2,4], default=None)
    args = parser.parse_args()
    if args.wait_for:
        if args.wait_for == args.dataset:
            raise ValueError('A training run cannot wait for itself')
        pending = ROOT/'models'/f'{args.dataset}_status.json'
        pending.parent.mkdir(exist_ok=True)
        pending.write_text(json.dumps(dict(dataset=args.dataset,state='queued',waiting_for=args.wait_for,epochs_requested=args.epochs)))
        deadline = time.time()+4*60*60
        while True:
            previous = ROOT/'models'/f'{args.wait_for}_status.json'
            progress = json.loads(previous.read_text()) if previous.exists() else {}
            if progress.get('state') == 'ready':
                break
            if progress.get('state') == 'failed' or time.time()>deadline:
                pending.write_text(json.dumps(dict(dataset=args.dataset,state='failed',error='Previous training did not finish successfully')))
                raise RuntimeError('Previous training did not finish successfully')
            time.sleep(10)
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requires CUDA-enabled PyTorch. See README.')
    data = ROOT / 'data/processed' / args.dataset / 'dataset.yaml'
    if not data.exists():
        raise FileNotFoundError('Prepare and audit the dataset first: ' + str(data))
    models = ROOT / 'models'
    models.mkdir(exist_ok=True)
    status_file = models / f'{args.dataset}_status.json'
    started = time.time()
    def write_status(state, **extra):
        status = dict(dataset=args.dataset, state=state, elapsed_seconds=round(time.time()-started),
                      epochs_requested=args.epochs, **extra)
        tmp = status_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2, default=str), encoding='utf-8')
        tmp.replace(status_file)
    write_status('starting')
    model = YOLO(args.resume or 'yolo11s-seg.pt')
    def epoch_callback(trainer):
        write_status('training', epoch=trainer.epoch + 1,
                     metrics={k:float(v) for k,v in trainer.metrics.items()})
    model.add_callback('on_fit_epoch_end', epoch_callback)
    try:
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=0, workers=args.workers, seed=42, deterministic=True,
                    patience=20, optimizer='AdamW', lr0=0.001, cos_lr=True,
                    close_mosaic=15, mosaic=0.5, mixup=0.0, degrees=180,
                    flipud=0.5, fliplr=0.5, scale=0.25, translate=0.1,
                    hsv_h=0.015, hsv_s=0.3, hsv_v=0.3,
                    mask_ratio=args.mask_ratio or (2 if args.dataset=='swiss' else 4), overlap_mask=True, cache=False, amp=True,
                    project=str(ROOT/'runs/segment'), name=args.dataset,
                    exist_ok=False, plots=True, save_period=10,
                    resume=bool(args.resume))
        best_path = Path(model.trainer.best)
        if not best_path.exists():
            raise RuntimeError('Training produced no best checkpoint')
        checkpoint = models / (args.dataset + '_best.pt')
        shutil.copy2(best_path, checkpoint)
        best = YOLO(checkpoint)
        metrics = best.val(data=str(data), split='test', device=0, imgsz=args.imgsz,
                           batch=args.batch, workers=args.workers, plots=True,
                           project=str(ROOT/'runs/evaluation'), name=args.dataset)
        report = {k:float(v) for k,v in metrics.results_dict.items()}
        (models/f'{args.dataset}_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        write_status('ready', checkpoint=str(checkpoint), test_metrics=report,
                     epochs_completed=model.trainer.epoch + 1)
    except BaseException as exc:
        write_status('failed', error=str(exc))
        raise

if __name__ == '__main__':
    main()
