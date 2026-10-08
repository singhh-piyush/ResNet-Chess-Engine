"""Train masked style/value heads on frozen game splits; save resumable runs."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from engine.encoding import board_to_tensor, POLICY_SIZE, PROMOTIONS
from engine.model import ChessModel
from training.split import prepare


class Positions(Dataset):
    def __init__(self,labels,split,partition):
        self.rows=[]
        for path in sorted(Path(labels).glob('*.jsonl')):
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            if rows and split['games'][rows[0]['game_id']]==partition:self.rows.extend(rows)
        if not self.rows:raise ValueError(f'No {partition} positions')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        return board_to_tensor(chess.Board(row['fen'])),row['policy'],np.float32(row['value'])


def losses(policy,value,targets,values):
    mask=targets>=0
    style=nn.functional.cross_entropy(policy[mask],targets[mask],label_smoothing=.1) if mask.any() else policy.sum()*0
    evaluation=nn.functional.mse_loss(value.flatten(),values)
    return style+5*evaluation


def initialize_from_baseline(model,baseline):
    state=model.state_dict()
    if set(state)!=set(baseline):raise ValueError('Baseline architecture keys differ')
    for key,value in baseline.items():
        if state[key].shape==value.shape:
            state[key]=value
        elif key in ('policy_head.5.weight','policy_head.5.bias') and value.shape[0]==4096:
            state[key][:4096]=value
        else:
            raise ValueError(f'Incompatible baseline parameter: {key}')
    for index,move in enumerate(PROMOTIONS,4096):
        old_index=move.from_square*64+move.to_square
        state['policy_head.5.weight'][index]=baseline['policy_head.5.weight'][old_index]
        state['policy_head.5.bias'][index]=baseline['policy_head.5.bias'][old_index]-(0 if move.promotion==chess.QUEEN else 2)
    model.load_state_dict(state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',default='data/labels');p.add_argument('--output',default='data/run')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=256);p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda':raise SystemExit('CUDA unavailable; run training where the GPU is accessible')
    split=prepare(labels=args.labels)
    loaders={name:DataLoader(Positions(args.labels,split,name),batch_size=args.batch,shuffle=name=='train',num_workers=2,pin_memory=True) for name in ('train','validation')}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    model=ChessModel().to(device)
    # Initialize shared representation from the verified old fold-0 baseline. The
    # promotion head is new; unrelated policy rows keep their existing weights.
    baseline=torch.load('models/legacy-cv0.pth',map_location=device,weights_only=True)
    initialize_from_baseline(model,baseline)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001,fused=True)
    scaler=torch.amp.GradScaler('cuda');start=0;best=float('inf');stale=0
    split_hash=hashlib.sha256(Path('data/split.json').read_bytes()).hexdigest()
    dataset_hash=hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(args.labels).glob('*.jsonl'))).encode()).hexdigest()
    if args.resume:
        saved=torch.load(output/'resume.pth',map_location=device,weights_only=True)
        if saved['split_sha256']!=split_hash or saved['dataset_sha256']!=dataset_hash:raise ValueError('Resume dataset differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start=saved['epoch']+1;best=saved['best'];stale=saved['stale'];torch.set_rng_state(saved['rng_cpu'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])
    for epoch in range(start,args.epochs):
        model.train();total=0
        for i,(inputs,targets,values) in enumerate(loaders['train']):
            inputs=inputs.to(device);targets=targets.to(device);values=values.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                policy,value=model(inputs);loss=losses(policy,value,targets,values)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1)
            scaler.step(optimizer);scaler.update();total+=loss.item()
            if i%100==0:print(f'Epoch {epoch+1} batch {i}/{len(loaders["train"])} loss {loss.item():.4f}',flush=True)
        model.eval();validation=0;correct=count=0;squared=0;n=0
        with torch.inference_mode():
            for inputs,targets,values in loaders['validation']:
                inputs=inputs.to(device);targets=targets.to(device);values=values.to(device)
                policy,value=model(inputs);validation+=losses(policy,value,targets,values).item()*len(inputs)
                mask=targets>=0;correct+=int((policy.argmax(1)[mask]==targets[mask]).sum());count+=int(mask.sum())
                squared+=float(((value.flatten()-values)**2).sum());n+=len(inputs)
        validation/=n
        metrics={'epoch':epoch+1,'validation_loss':validation,'style_top1':correct/max(count,1),'value_mse':squared/n}
        print(json.dumps(metrics),flush=True)
        improved=validation<best;best=min(best,validation);stale=0 if improved else stale+1
        saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'stale':stale,
               'split_sha256':split_hash,'dataset_sha256':dataset_hash,'metrics':metrics,'policy_size':POLICY_SIZE,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
        torch.save(saved,output/'resume.tmp');(output/'resume.tmp').replace(output/'resume.pth')
        if improved:torch.save({'model':model.state_dict(),'metrics':metrics,'split_sha256':split_hash,'dataset_sha256':dataset_hash},output/'best.pth')
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        if stale>=5:break
    print('Training complete; candidate still requires independent quality evaluation.',flush=True)

if __name__=='__main__':main()
