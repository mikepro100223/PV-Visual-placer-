"""Explicit GPU retraining entry point for the user's local datasets.

Run manually after reviewing data/reports/*_audit.json. This file does not
launch training on import and is not registered as a background task.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--prepared',action='store_true',help='Use already audited prepared manifests')
    parser.add_argument('--resume-rid',type=Path,default=None)
    args=parser.parse_args()
    source=ROOT/'datasets/kaggle-swiss-solar-panels-segmentation'
    if not source.exists():source=ROOT/'datasets'
    if not args.prepared:
        run('scripts/prepare_swiss.py','--source',str(source),'--task','swiss')
        run('scripts/prepare_swiss.py','--source',str(source),'--task','roof')
        if (ROOT/'datasets/SOLKAT_DACH.gpkg').exists():run('scripts/prepare_geneva_roofs.py')
        run('scripts/prepare_rid.py')
    if args.prepare_only:return
    # Separate processes release CUDA allocations between models. Obstacle and
    # roof models are trained first because the existing PV model is available.
    for name,batch in [('rid',4),('roof',4),('swiss',6)]:
        extra=['--resume',str(args.resume_rid)] if name=='rid' and args.resume_rid else []
        run('scripts/train.py','--dataset',name,'--epochs','100','--imgsz','1024','--batch',str(batch),'--workers','2',*extra)

if __name__=='__main__':main()
