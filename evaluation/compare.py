"""Freeze an independent test suite and gate model promotion with offline Stockfish."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import chess
import chess.engine
import numpy as np
from engine.encoding import encode_move
from engine.runtime import Runtime
from engine.search import Search


def suite(labels,split_path,path,count):
    path=Path(path)
    if path.exists():return [json.loads(line) for line in path.read_text().splitlines()]
    split=json.loads(Path(split_path).read_text())['games'];games=[]
    for source in sorted(Path(labels).glob('*.jsonl')):
        rows=[json.loads(line) for line in source.read_text().splitlines()]
        if not rows or split[rows[0]['game_id']]!='test':continue
        candidates=[r for r in rows if r['policy']>=0 and r['ply']>=8 and not chess.Board(r['fen']).is_game_over(claim_draw=True)]
        if candidates:
            candidates.sort(key=lambda r:hashlib.sha256((r['game_id']+str(r['ply'])).encode()).hexdigest())
            games.append(candidates)
    games.sort(key=lambda rows:hashlib.sha256(rows[0]['game_id'].encode()).hexdigest())
    result=[];offset=0
    while len(result)<count:
        found=False
        for rows in games:
            if offset<len(rows):result.append(rows[offset]);found=True
            if len(result)==count:break
        if not found:raise ValueError(f'Only {len(result)} eligible heldout positions; {count} required')
        offset+=1
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(row)+'\n' for row in result))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',default='models/release.json')
    p.add_argument('--suite',default='data/evaluation/suite.jsonl');p.add_argument('--output',default='data/evaluation/results.jsonl')
    p.add_argument('--count',type=int,default=500);p.add_argument('--seconds',type=float,default=9.5);p.add_argument('--nodes',type=int,default=200000)
    args=p.parse_args()
    if args.count<500:raise ValueError('Release evaluation requires at least 500 heldout positions')
    rows=suite('data/labels','data/split.json',args.suite,args.count)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    config={'baseline_sha256':hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
            'suite_sha256':hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),'seconds':args.seconds,'nodes':args.nodes}
    config_path=output.with_suffix('.config.json')
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Evaluation configuration changed; use a new output')
    config_path.write_text(json.dumps(config,indent=2))
    done={r['key']:r for r in [json.loads(line) for line in output.read_text().splitlines()]} if output.exists() else {}
    runtimes={'baseline':Runtime(args.baseline),'candidate':Runtime(args.candidate)}
    with chess.engine.SimpleEngine.popen_uci('/usr/bin/stockfish') as teacher:
        teacher.configure({'Threads':1,'Hash':128})
        def score(board):
            teacher.configure({'Clear Hash':None})
            return teacher.analyse(board,chess.engine.Limit(nodes=args.nodes))['score'].pov(board.turn).score(mate_score=10000)
        for i,row in enumerate(rows):
            key=f'{row["game_id"]}:{row["ply"]}'
            if key in done:continue
            board=chess.Board(row['fen']);best=score(board);result={'key':key}
            for name,runtime in runtimes.items():
                policy,_=runtime.evaluate([board])[0]
                legal=list(board.legal_moves);ranked=sorted(legal,key=lambda m:runtime.logits(board,policy,[m])[0],reverse=True)
                start=time.monotonic();chosen=Search(runtime,seconds=args.seconds).run(board);elapsed=time.monotonic()-start
                move=chess.Move.from_uci(chosen['move']);child=board.copy();child.push(move)
                post=score(child);loss=max(0,best+post)
                result[name]={'loss_cp':loss,'blunder':loss>150,'style_top1':ranked[0].uci()==row['played_move'],
                              'style_rank':next(j+1 for j,m in enumerate(ranked) if m.uci()==row['played_move']),
                              'move':move.uci(),'seconds':elapsed}
            with output.open('a') as f:f.write(json.dumps(result)+'\n')
            done[key]=result;print(f'{i+1}/{len(rows)}',flush=True)
    reports={}
    for name in runtimes:
        values=[r[name] for r in done.values()]
        reports[name]={'blunder_rate':float(np.mean([v['blunder'] for v in values])), 'style_top1':float(np.mean([v['style_top1'] for v in values])),
                       'mean_loss_cp':float(np.mean([v['loss_cp'] for v in values])),'latency_p95':float(np.percentile([v['seconds'] for v in values],95))}
    base,candidate=reports['baseline'],reports['candidate']
    reports['positions']=len(done)
    reports['passed']=len(done)>=500 and base['blunder_rate']>0 and candidate['blunder_rate']<=base['blunder_rate']*.8 and candidate['style_top1']>=base['style_top1']-.05 and candidate['latency_p95']<=10
    reports['configuration']=config
    output.with_suffix('.summary.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2),flush=True)
    if not reports['passed']:raise SystemExit('Quality gate failed; retain the current release')

if __name__=='__main__':main()
