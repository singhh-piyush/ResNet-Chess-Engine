"""GPU-resident training on packed positions, shared by single runs, K-fold teachers and students.

The whole packed dataset (~250 MB) lives in GPU memory: batches are sliced on the device,
so no CPU loader workers or per-batch host copies are involved.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import chess
import numpy as np
import torch
from torch import nn
from engine.encoding import POLICY_SIZE, PROMOTIONS
from engine.model import ChessModel

LEGACY = 'models/legacy-cv0.pth'
LEGACY_ARCHITECTURE = {'num_res_blocks':15, 'channels':192}
DEFAULTS = {'architecture':LEGACY_ARCHITECTURE, 'init':'legacy', 'epochs':6, 'batch':1024, 'lr_tower':3e-5, 'lr_head':1e-4,
            'weight_decay':1e-4, 'warmup':.05, 'block_dropout':0., 'head_dropout':.5, 'label_smoothing':.1, 'value_weight':5.,
            'half_life':None, 'policy':'filtered', 'blunder_weight':.3, 'alpha':1., 'opponent_weight':.5,
            'evals_per_epoch':4, 'patience':4, 'compile':True}
SECONDS_PER_MONTH = 30.44*86400


def load_packed(path='data/packed', device='cuda'):
    path = Path(path)
    meta = json.loads((path/'meta.json').read_text())
    host = {name:np.load(path/f'{name}.npy') for name in ('planes','move_number','policy','played','user','value','ply','game','end_time','time_class','partition')}
    data = {'meta':meta, 'host':host}
    for name in ('planes','move_number','policy','played','user','value'):
        array = host[name].astype(np.int64) if name in ('policy','played') else host[name]
        data[name] = torch.from_numpy(array).to(device)
    data['device'] = device
    return data


def batch_inputs(data, index):
    """Rebuild the 19-plane float input on the GPU (plane 18 is the saturating move number)."""
    planes = data['planes'][index].float()
    move = (data['move_number'][index].float()/60).view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
    return torch.cat([planes, move], 1).contiguous(memory_format=torch.channels_last)


def losses(policy, value, targets, values, weights=None, soft_index=None, soft_prob=None, soft_weights=None,
           alpha=1., value_weight=5., label_smoothing=.1):
    """Weighted style cross-entropy on your moves (target -1 = no label), optional distillation
    toward soft teacher probabilities, and value regression for every position."""
    weights = torch.ones_like(values) if weights is None else weights
    mask = targets >= 0
    if mask.any():
        ce = nn.functional.cross_entropy(policy[mask].float(), targets[mask], label_smoothing=label_smoothing, reduction='none')
        style = (ce*weights[mask]).sum()/weights[mask].sum()
    else:
        style = policy.sum()*0
    loss = alpha*style
    if soft_index is not None and alpha < 1:
        log_probs = torch.log_softmax(policy.float(), 1).gather(1, soft_index)
        distill = -(soft_prob*log_probs).sum(1)
        loss = loss+(1-alpha)*(distill*soft_weights).sum()/soft_weights.sum().clamp_min(1e-6)
    evaluation = (((value.flatten().float()-values)**2)*weights).sum()/weights.sum()
    return loss+value_weight*evaluation


def initialize_from_baseline(model, baseline):
    state = model.state_dict()
    if set(state) != set(baseline): raise ValueError('Baseline architecture keys differ')
    for key, value in baseline.items():
        if state[key].shape == value.shape:
            state[key] = value
        elif key in ('policy_head.5.weight', 'policy_head.5.bias') and value.shape[0] == 4096:
            state[key][:4096] = value
        else:
            raise ValueError(f'Incompatible baseline parameter: {key}')
    for index, move in enumerate(PROMOTIONS, 4096):
        old_index = move.from_square*64+move.to_square
        state['policy_head.5.weight'][index] = baseline['policy_head.5.weight'][old_index]
        state['policy_head.5.bias'][index] = baseline['policy_head.5.bias'][old_index]-(0 if move.promotion == chess.QUEEN else 2)
    model.load_state_dict(state)


def build_model(config, device):
    model = ChessModel(policy_size=POLICY_SIZE, block_dropout=config['block_dropout'], head_dropout=config['head_dropout'], **config['architecture'])
    if config['init'] == 'legacy':
        initialize_from_baseline(model, torch.load(LEGACY, map_location='cpu', weights_only=True))
    elif config['init'] != 'scratch':
        saved = torch.load(config['init'], map_location='cpu', weights_only=True)
        model.load_state_dict(saved.get('model', saved))
    return model.to(device).to(memory_format=torch.channels_last)


def targets_and_weights(data, config):
    """Per-position style target and weight for the chosen policy variant and recency half-life."""
    if config['policy'] == 'all':
        targets = torch.where(data['user'] & (data['played'] >= 0), data['played'], torch.full_like(data['played'], -1))
        weights = torch.where(data['policy'] >= 0, 1., config['blunder_weight'])
    else:
        targets = data['policy']
        weights = torch.ones_like(data['value'])
    end_time = torch.from_numpy(data['host']['end_time']).to(data['device'])
    if config['half_life']:
        age = (end_time.max()-end_time).double()/SECONDS_PER_MONTH
        weights = weights*torch.pow(.5, age/config['half_life']).float()
    return targets, weights.float()


def evaluate(model, data, index, batch=4096, recent=None):
    """Style metrics always use the filtered target (your moves losing at most 150 cp),
    so every training variant is scored on the same question."""
    model.eval(); totals = {'ce':0., 'top1':0, 'top3':0, 'n':0, 'value_se':0., 'positions':0}
    recent_hits = recent_n = 0
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for start in range(0, len(index), batch):
            chunk = index[start:start+batch]
            policy, value = model(batch_inputs(data, chunk))
            policy = policy.float(); targets = data['policy'][chunk]; mask = targets >= 0
            totals['value_se'] += float(((value.flatten().float()-data['value'][chunk])**2).sum()); totals['positions'] += len(chunk)
            if mask.any():
                top = policy[mask].topk(3, 1).indices; hit1 = top[:, 0] == targets[mask]
                totals['ce'] += float(nn.functional.cross_entropy(policy[mask], targets[mask], reduction='sum'))
                totals['top1'] += int(hit1.sum()); totals['top3'] += int((top == targets[mask, None]).any(1).sum()); totals['n'] += int(mask.sum())
                if recent is not None:
                    keep = recent[chunk][mask]; recent_hits += int(hit1[keep].sum()); recent_n += int(keep.sum())
    n = max(totals['n'], 1)
    metrics = {'style_ce':totals['ce']/n, 'style_top1':totals['top1']/n, 'style_top3':totals['top3']/n,
               'value_mse':totals['value_se']/max(totals['positions'], 1), 'style_positions':totals['n']}
    if recent is not None:
        metrics.update(recent_top1=recent_hits/max(recent_n, 1), recent_positions=recent_n)
    return metrics


def recent_mask(data, months=24):
    end_time = torch.from_numpy(data['host']['end_time']).to(data['device'])
    return end_time >= end_time.max()-months*SECONDS_PER_MONTH


def train_one(data, config, train_index, val_index=None, soft=None, log=print, steps=None):
    """Train one model. With `val_index`, early-stop on validation style CE and return the
    best weights; without, train for exactly `steps` optimizer steps (or the configured epochs)."""
    config = {**DEFAULTS, **config}
    device = data['device']
    torch.manual_seed(42)
    model = build_model(config, device)
    targets, weights = targets_and_weights(data, config)
    heads = [p for name, p in model.named_parameters() if name.startswith(('policy_head', 'value_head'))]
    tower = [p for name, p in model.named_parameters() if not name.startswith(('policy_head', 'value_head'))]
    optimizer = torch.optim.AdamW([{'params':tower, 'lr':config['lr_tower']}, {'params':heads, 'lr':config['lr_head']}],
                                  weight_decay=config['weight_decay'], fused=True)
    per_epoch = math.ceil(len(train_index)/config['batch'])
    total = steps or per_epoch*config['epochs']
    warmup = max(1, int(total*config['warmup']))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1, (s+1)/warmup)*.5*(1+math.cos(math.pi*min(s, total)/total)))
    forward = torch.compile(model) if config['compile'] else model
    recent = recent_mask(data)
    every = max(1, per_epoch//config['evals_per_epoch'])
    best = {'style_ce':math.inf}; best_state = None; stale = 0; history = []; step = 0; started = time.monotonic()
    while step < total:
        order = train_index[torch.randperm(len(train_index), device=device)]
        for start in range(0, len(order), config['batch']):
            if step >= total: break
            index = order[start:start+config['batch']]
            model.train()
            with torch.autocast('cuda', torch.bfloat16):
                policy, value = forward(batch_inputs(data, index))
            kwargs = {}
            if soft is not None:
                # Opponent-to-move positions only teach through the teacher, at reduced weight.
                kwargs = {'soft_index':soft['index'][index], 'soft_prob':soft['prob'][index].float(),
                          'soft_weights':torch.where(data['user'][index], 1., config['opponent_weight'])*weights[index]}
            values = data['value'][index] if soft is None else .5*data['value'][index]+.5*soft['value'][index].float()
            loss = losses(policy, value, targets[index], values, weights[index], alpha=config['alpha'],
                          value_weight=config['value_weight'], label_smoothing=config['label_smoothing'], **kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step(); schedule.step(); step += 1
            if step % every == 0 or step == total:
                rate = step*config['batch']/(time.monotonic()-started)
                entry = {'step':step, 'epoch':round(step/per_epoch, 2), 'loss':float(loss), 'samples_per_s':round(rate)}
                if val_index is not None:
                    entry.update(evaluate(model, data, val_index, recent=recent))
                    if entry['style_ce'] < best['style_ce']:
                        best = entry; best_state = copy.deepcopy(model.state_dict()); stale = 0
                    else:
                        stale += 1
                history.append(entry); log(json.dumps(entry))
                if val_index is not None and stale >= config['patience']:
                    step = total
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict()); best = history[-1] if history else {}
    model.load_state_dict(best_state)
    return model, best, history


@torch.inference_mode()
def predict(model, data, index, top=16, batch=4096):
    """Soft policy targets (top-k probabilities) and values for distillation."""
    model.eval(); indices = []; probs = []; values = []
    with torch.autocast('cuda', torch.bfloat16):
        for start in range(0, len(index), batch):
            policy, value = model(batch_inputs(data, index[start:start+batch]))
            p = torch.softmax(policy.float(), 1).topk(top, 1)
            # Renormalise the kept mass so each soft target is a distribution.
            indices.append(p.indices); probs.append((p.values/p.values.sum(1, keepdim=True)).half()); values.append(value.flatten().half())
    return torch.cat(indices), torch.cat(probs), torch.cat(values)


def save(model, config, metrics, path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model':model.state_dict(), 'config':config, 'metrics':metrics,
                'dataset_sha256':data['meta']['dataset_sha256'], 'split_sha256':data['meta']['split_sha256']}, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--packed', default='data/packed'); p.add_argument('--output', default='data/run2')
    p.add_argument('--config', default='{}', help='JSON overrides of training.train.DEFAULTS')
    args = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable; run training where the GPU is accessible')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    data = load_packed(args.packed)
    partition = torch.from_numpy(data['host']['partition']).cuda()
    train = torch.nonzero(partition == 0).flatten(); val = torch.nonzero(partition == 1).flatten()
    config = {**DEFAULTS, **json.loads(args.config)}
    model, best, _ = train_one(data, config, train, val)
    save(model, config, best, Path(args.output)/'best.pth', data)
    print('Training complete; candidate still requires independent quality evaluation.', json.dumps(best), flush=True)

if __name__ == '__main__': main()
