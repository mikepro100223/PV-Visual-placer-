"""Train a full-resolution semantic detector, preserving small Geneva objects."""
import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from backend.services.obstacle_network import ObstacleUNet
from training.evaluate_pixels import scores


class Chips(Dataset):
    def __init__(self, root, split):
        self.split = split
        self.items = []
        for path in sorted((Path(root)/'images'/split).glob('*.jpg')):
            image = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            mask = np.zeros(image.shape[:2], np.uint8)
            for line in (Path(root)/'labels'/split/(path.stem+'.txt')).read_text().splitlines():
                points = np.array(list(map(float,line.split()[1:]))).reshape(-1,2)*[image.shape[1],image.shape[0]]
                cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
            self.items.append((image, mask, path.name))

    def __len__(self):
        return len(self.items)*(2 if self.split=='train' else 1)

    def __getitem__(self, index):
        image, mask, name = self.items[index % len(self.items)]
        if self.split=='train':
            # Half random crops, half centred near a labelled object. The
            # held-out sets retain all background and are never resampled.
            y,x = random.randrange(321),random.randrange(321)
            if random.random()<.5 and mask.any():
                yy,xx=np.nonzero(mask);i=random.randrange(len(yy))
                y=int(np.clip(yy[i]-random.randrange(64,256),0,320))
                x=int(np.clip(xx[i]-random.randrange(64,256),0,320))
            image,mask=image[y:y+320,x:x+320],mask[y:y+320,x:x+320]
            k=random.randrange(4);image=np.rot90(image,k);mask=np.rot90(mask,k)
            if random.random()<.5: image,mask=image[:,::-1],mask[:,::-1]
        image=np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)/255
        if self.split=='train':image=np.clip(image*random.uniform(.8,1.2),0,1)
        return torch.from_numpy(image),torch.from_numpy(np.ascontiguousarray(mask[None],dtype=np.float32)),name


def validate(model, loader, device, thresholds):
    model.eval();counts=np.zeros((len(thresholds),3),np.int64)
    with torch.inference_mode():
        for image,truth,_ in loader:
            probability=model(image.to(device)).sigmoid().cpu().numpy()
            target=truth.numpy()>.5
            for i,threshold in enumerate(thresholds):
                predicted=probability>=threshold
                counts[i]+=[np.count_nonzero(predicted&target),np.count_nonzero(predicted&~target),np.count_nonzero(~predicted&target)]
    return [{'threshold':t,**scores(*map(int,c)),'counts':list(map(int,c))} for t,c in zip(thresholds,counts)]


def main(args):
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ObstacleUNet().to(device)
    train=DataLoader(Chips(args.data,'train'),batch_size=12,shuffle=True,num_workers=0)
    val=DataLoader(Chips(args.data,'val'),batch_size=2,num_workers=0)
    optimiser=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    best=-1.;history=[]
    for epoch in range(args.epochs):
        model.train();losses=[]
        for image,target,_ in train:
            image,target=image.to(device),target.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                logits=model(image)
                bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(15.,device=device))
                probability=logits.sigmoid()
                dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
                loss=bce+dice
            scaler.scale(loss).backward();scaler.step(optimiser);scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics=validate(model,val,device,[.25,.4,.5,.6,.75,.9])
        choice=max(metrics,key=lambda m:m['iou'] or 0)
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':choice}
        history.append(record)
        if choice['iou']>best:
            best=choice['iou']
            torch.save({'state_dict':model.state_dict(),'architecture':'ObstacleUNet-v1','threshold':choice['threshold'],
                'validation':choice,'epoch':epoch+1,'target':'Geneva surveyed superstructure footprint'},root/'best.pt')
        (root/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/geneva_clean')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--output',default='runs/obstacle_unet')
    main(parser.parse_args())
