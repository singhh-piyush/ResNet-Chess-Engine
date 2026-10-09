"""K-fold by game over the train+validation pool, then distillation into one fast student.

Stages (each resumable, all on the GPU, the frozen test split is never read):
  sweep     every config in GRID x every fold -> data/kfold/sweep.jsonl
  teachers  best config x every fold; each teacher labels only its held-out fold, giving
            out-of-fold (OOF) soft policy/value targets for every pool position
  student   --cv-fold k: train on the other folds with OOF targets, score on fold k
            --final:     train on the whole pool for the CV-chosen step count, write a manifest
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from training.train import DEFAULTS, LEGACY_ARCHITECTURE, build_model, load_packed, predict, save, train_one

GRID = {
    'base':{},
    'half_life_24':{'half_life':24},
    'half_life_12':{'half_life':12},
    'all_moves':{'policy':'all'},
    'all_moves_half_life_12':{'policy':'all', 'half_life':12},
    'lr_x3':{'lr_tower':1e-4, 'lr_head':3e-4},
    'lr_div3':{'lr_tower':1e-5, 'lr_head':3e-5},
    'block_dropout_0.2':{'block_dropout':.2},
}
STUDENT = {'init':'scratch', 'epochs':30, 'lr_tower':1e-3, 'lr_head':1e-3, 'alpha':.5, 'head_dropout':.3, 'patience':8}


def folds(data, count, output):
    """Deterministic game-level folds: pool games sorted by hash, dealt round-robin."""
    path = Path(output)/'folds.json'
    games = data['meta']['games']; partition = data['host']['partition']; game = data['host']['game']
    pool = sorted({int(g) for g, p in zip(game, partition) if p in (0, 1)}, key=lambda g:hashlib.sha256(games[g].encode()).hexdigest())
    assignment = {games[g]:i % count for i, g in enumerate(pool)}
    if path.exists() and json.loads(path.read_text()) != assignment: raise ValueError('Fold assignment changed; use a new output directory')
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(assignment)+'\n')
    fold_of_game = torch.full((len(games),), -1, dtype=torch.long)
    for game_id, fold in assignment.items(): fold_of_game[games.index(game_id)] = fold
    return fold_of_game[torch.from_numpy(game).long()].to(data['device'])


def split(fold_of_position, fold):
    train = torch.nonzero((fold_of_position >= 0) & (fold_of_position != fold)).flatten()
    return train, torch.nonzero(fold_of_position == fold).flatten()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def best_config(output):
    rows = records(Path(output)/'sweep.jsonl'); names = {r['name'] for r in rows}
    complete = [n for n in names if len([r for r in rows if r['name'] == n]) == max(r['fold'] for r in rows)+1]
    if not complete: raise SystemExit('Run --stage sweep first')
    score = lambda n: sum(r['metrics']['recent_top1'] for r in rows if r['name'] == n)
    name = max(complete, key=score)
    return name, GRID[name]


def report_sweep(output):
    rows = records(Path(output)/'sweep.jsonl')
    for name in GRID:
        mine = [r['metrics'] for r in rows if r['name'] == name]
        if not mine: continue
        mean = lambda k: sum(m[k] for m in mine)/len(mine)
        spread = lambda k: (sum((m[k]-mean(k))**2 for m in mine)/len(mine))**.5
        print(f'{name:24s} folds={len(mine)} recent_top1={mean("recent_top1"):.4f}±{spread("recent_top1"):.4f} '
              f'top1={mean("style_top1"):.4f} top3={mean("style_top3"):.4f} value_mse={mean("value_mse"):.4f} epoch={mean("epoch"):.1f}', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True, choices=('sweep', 'teachers', 'student'))
    p.add_argument('--packed', default='data/packed'); p.add_argument('--output', default='data/kfold'); p.add_argument('--folds', type=int, default=5)
    p.add_argument('--only', help='comma-separated GRID names for the sweep'); p.add_argument('--arch', default='{"num_res_blocks":10,"channels":128,"policy_channels":16}')
    p.add_argument('--cv-fold', type=int); p.add_argument('--final', action='store_true'); p.add_argument('--name', help='student name, e.g. 10x128')
    args = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    data = load_packed(args.packed)
    fold_of_position = folds(data, args.folds, output)
    log = lambda line: print(line, flush=True)
    if args.stage == 'sweep':
        done = {(r['name'], r['fold']) for r in records(output/'sweep.jsonl')}
        for name in (args.only.split(',') if args.only else GRID):
            for fold in range(args.folds):
                if (name, fold) in done: continue
                log(f'=== sweep {name} fold {fold}')
                train, val = split(fold_of_position, fold)
                _, best, history = train_one(data, GRID[name], train, val, log=log)
                with (output/'sweep.jsonl').open('a') as f: f.write(json.dumps({'name':name, 'fold':fold, 'metrics':best, 'history':history})+'\n')
        report_sweep(output)
    elif args.stage == 'teachers':
        name, config = best_config(output); log(f'Teacher config: {name} {json.dumps(config)}')
        positions = len(data['value'])
        soft = {'index':torch.zeros(positions, 16, dtype=torch.long, device='cuda'), 'prob':torch.zeros(positions, 16, dtype=torch.half, device='cuda'),
                'value':torch.zeros(positions, dtype=torch.half, device='cuda')}
        summary = []
        for fold in range(args.folds):
            train, val = split(fold_of_position, fold)
            path = output/f'teacher_{fold}.pth'
            if path.exists():
                model = build_model({**DEFAULTS, **config, 'init':str(path)}, 'cuda')
                best = torch.load(path, map_location='cpu', weights_only=True)['metrics']
            else:
                log(f'=== teacher fold {fold}')
                model, best, _ = train_one(data, config, train, val, log=log)
                save(model, {**DEFAULTS, **config}, best, path, data)
            # Each teacher labels only the fold it never trained on.
            soft['index'][val], soft['prob'][val], soft['value'][val] = predict(model, data, val)
            summary.append(best); del model; torch.cuda.empty_cache()
        torch.save({k:v.cpu() for k, v in soft.items()} | {'config':name}, output/'oof.pt')
        mean = lambda k: sum(m[k] for m in summary)/len(summary)
        log(json.dumps({'teacher_config':name, 'oof_top1':mean('style_top1'), 'oof_recent_top1':mean('recent_top1'), 'oof_value_mse':mean('value_mse')}))
    else:
        name, teacher = best_config(output)
        saved = torch.load(output/'oof.pt', weights_only=True); soft = {k:saved[k].cuda() for k in ('index', 'prob', 'value')}
        architecture = json.loads(args.arch)
        tag = args.name or f'{architecture["num_res_blocks"]}x{architecture["channels"]}'
        # Same-size fallback keeps the verified legacy initialization.
        legacy = {'policy_channels':32, **architecture} == {'policy_channels':32, **LEGACY_ARCHITECTURE}
        config = {**teacher, **STUDENT, 'architecture':architecture}
        if legacy: config.update(init='legacy', lr_tower=teacher.get('lr_tower', DEFAULTS['lr_tower']), lr_head=teacher.get('lr_head', DEFAULTS['lr_head']), epochs=DEFAULTS['epochs'], head_dropout=DEFAULTS['head_dropout'])
        if args.final:
            cv = [r for r in records(output/'students.jsonl') if r['name'] == tag]
            if not cv: raise SystemExit(f'Run --stage student --cv-fold 0 --name {tag} first')
            pool = torch.nonzero(fold_of_position >= 0).flatten()
            # The CV run trained on (k-1)/k of the pool; scale its best step count to the full pool.
            steps = round(cv[-1]['metrics']['step']*args.folds/(args.folds-1))
            log(f'=== student {tag} final, {steps} steps')
            model, _, history = train_one(data, config, pool, None, soft=soft, log=log, steps=steps)
            weights = Path('models')/f'student-{tag}.pth'
            torch.save({'model':model.state_dict()}, weights)
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            manifest = {'path':weights.name, 'sha256':digest, 'policy_size':4272, 'version':f'student-{tag}-{digest[:12]}',
                        'architecture':architecture, 'quality_status':'pending independent gate',
                        'training':{'teacher_config':name, 'cv_metrics':cv[-1]['metrics'], 'steps':steps}}
            (Path('models')/f'student-{tag}.json').write_text(json.dumps(manifest, indent=2)+'\n')
            log(f'Wrote models/student-{tag}.json')
        else:
            fold = args.cv_fold if args.cv_fold is not None else 0
            train, val = split(fold_of_position, fold)
            log(f'=== student {tag} cv fold {fold}')
            _, best, history = train_one(data, config, train, val, soft=soft, log=log)
            with (output/'students.jsonl').open('a') as f: f.write(json.dumps({'name':tag, 'fold':fold, 'architecture':architecture, 'metrics':best})+'\n')
            log(json.dumps({'student':tag, **best}))

if __name__ == '__main__': main()
