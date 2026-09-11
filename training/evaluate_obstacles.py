"""Evaluate the frozen semantic checkpoint without selecting on test images."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from backend.services.obstacle_network import ObstacleUNet
from training.train_obstacles import Chips, validate


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='runs/obstacle_unet/best.pt')
    parser.add_argument('--data',default='data/geneva_clean')
    parser.add_argument('--output',default='models/geneva-unet-evaluation.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=torch.load(args.model,map_location='cpu',weights_only=True)
    model=ObstacleUNet().to(device);model.load_state_dict(checkpoint['state_dict'])
    dataset=Chips(args.data,'test')
    metrics=validate(model,DataLoader(dataset,batch_size=2),device,[checkpoint['threshold']])[0]
    report={'model_sha256':hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        'architecture':checkpoint['architecture'],'epoch':checkpoint['epoch'],
        'validation':checkpoint['validation'],'test':metrics,'test_chips':len(dataset),
        'deployment':'research_only_not_enabled',
        'reason':'Insufficient validation precision and IoU; retain surveyed and DSM obstacles in the application.',
        'limitations':'Incomplete cadastral target and displacement/age mismatch with SWISSIMAGE. Does not measure exhaustive obstacle accuracy.'}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
