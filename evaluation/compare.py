"""Freeze an independent test suite and gate model promotion with offline Stockfish.

Two runs, in order:
  --latency-only  CPU with 2 threads (mirrors the HF Space): wall-clock latency and the
                  number of search nodes each model reaches within the serving budget.
  (default)       GPU: each model searches with exactly that node budget, so quality is
                  measured at serving strength without serving wall-clock cost. Stockfish
                  scoring runs in a CPU process pool and is cached across reruns.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
import time
from pathlib import Path
import chess
import chess.engine
import numpy as np
from engine.runtime import Runtime
from engine.search import Search

ENGINE=None
NODES=None


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


def init(stockfish,nodes):
    global ENGINE,NODES
    ENGINE=chess.engine.SimpleEngine.popen_uci(stockfish);ENGINE.configure({'Threads':1,'Hash':128});NODES=nodes


def stockfish_score(fen):
    board=chess.Board(fen);outcome=board.outcome(claim_draw=True)
    if outcome:return fen,0 if outcome.winner is None else (10000 if outcome.winner==board.turn else -10000)
    # Clear hash so every score is reproducible regardless of worker scheduling.
    ENGINE.configure({'Clear Hash':None})
    return fen,ENGINE.analyse(board,chess.engine.Limit(nodes=NODES))['score'].pov(board.turn).score(mate_score=10000)


def score_all(fens,cache_path,stockfish,nodes,workers):
    cache_path=Path(cache_path)
    cache={r['fen']:r['cp'] for r in map(json.loads,cache_path.read_text().splitlines())} if cache_path.exists() else {}
    todo=sorted(set(fens)-set(cache))
    if todo:
        print(f'Stockfish: scoring {len(todo)} positions with {workers} workers',flush=True)
        with multiprocessing.Pool(workers,initializer=init,initargs=(stockfish,nodes)) as pool, cache_path.open('a') as f:
            for i,(fen,cp) in enumerate(pool.imap_unordered(stockfish_score,todo,chunksize=4),1):
                cache[fen]=cp;f.write(json.dumps({'fen':fen,'cp':cp})+'\n')
                if i%200==0:f.flush();print(f'Stockfish: {i}/{len(todo)}',flush=True)
    return cache


def policy_ranks(runtime,rows,batch=512):
    ranks=[]
    for start in range(0,len(rows),batch):
        chunk=rows[start:start+batch];boards=[chess.Board(r['fen']) for r in chunk]
        for (policy,_),board,row in zip(runtime.evaluate(boards),boards,chunk):
            legal=list(board.legal_moves);logits=runtime.logits(board,policy,legal)
            ranked=[legal[i].uci() for i in np.argsort(-logits,kind='stable')]
            ranks.append(ranked.index(row['played_move'])+1)
        runtime.cache.clear()
    return ranks


def latency(args,rows):
    report={}
    for name,manifest in (('baseline',args.baseline),('candidate',args.candidate)):
        runtime=Runtime(manifest,device='cpu',threads=2);runtime.evaluate([chess.Board()])
        seconds=[];nodes=[];depths=[]
        for row in rows[:args.latency_count]:
            board=chess.Board(row['fen']);runtime.cache.clear();start=time.monotonic()
            result=Search(runtime,seconds=args.seconds,hard_seconds=args.hard_seconds,rng=np.random.default_rng(0)).run(board)
            seconds.append(time.monotonic()-start)
            if not result['book']:nodes.append(result['positions']);depths.append(result['depth'])
        report[name]={'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                      'latency_p50':float(np.percentile(seconds,50)),'latency_p95':float(np.percentile(seconds,95)),
                      'nodes_p50':int(np.median(nodes)),'depth_mean':float(np.mean(depths))}
        print(name,json.dumps(report[name]),flush=True)
    report['budget']={'seconds':args.seconds,'hard_seconds':args.hard_seconds,'positions':args.latency_count,'threads':2}
    return report


def bootstrap(differences,samples=5000):
    rng=np.random.default_rng(0);differences=np.asarray(differences,dtype=np.float64)
    means=differences[rng.integers(0,len(differences),(samples,len(differences)))].mean(1)
    return [float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]


def summarize(rows,results,latency_report,config):
    recent=max(r['metadata']['end_time'] for r in rows)-365*86400
    slices={'all':lambda r:True,'recent_12_months':lambda r:r['metadata']['end_time']>=recent,
            'ply_8_20':lambda r:r['ply']<20,'ply_20_40':lambda r:20<=r['ply']<40,'ply_40_plus':lambda r:r['ply']>=40}
    report={'positions':len(rows),'configuration':config}
    for name in ('baseline','candidate'):
        report[name]={}
        for label,keep in slices.items():
            values=[results[name][i] for i,r in enumerate(rows) if keep(r)]
            if not values:continue
            report[name][label]={'n':len(values),'blunder_rate':float(np.mean([v['blunder'] for v in values])),
                'style_top1':float(np.mean([v['rank']==1 for v in values])),'style_top3':float(np.mean([v['rank']<=3 for v in values])),
                'search_match':float(np.mean([v['match'] for v in values])),'mean_loss_cp':float(np.mean([v['loss_cp'] for v in values])),
                'book_rate':float(np.mean([v['book'] for v in values]))}
        report[name]['latency']=latency_report.get(name)
    base,cand=results['baseline'],results['candidate']
    blunder_ci=bootstrap([c['blunder']-b['blunder'] for b,c in zip(base,cand)])
    style_ci=bootstrap([(c['rank']==1)-(b['rank']==1) for b,c in zip(base,cand)])
    base_rate=report['baseline']['all']['blunder_rate'];cand_rate=report['candidate']['all']['blunder_rate']
    p95=(latency_report.get('candidate') or {}).get('latency_p95',math.inf)
    report['gate']={'blunder_delta_ci95':blunder_ci,'style_top1_delta_ci95':style_ci,'candidate_latency_p95':p95,
        'fewer_blunders':blunder_ci[1]<0 or cand_rate<=base_rate*.85,'style_kept':style_ci[0]>=-.02,'fast_enough':p95<=3.0}
    report['passed']=len(rows)>=500 and all(report['gate'][k] for k in ('fewer_blunders','style_kept','fast_enough'))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',default='models/release.json')
    p.add_argument('--suite',default='data/evaluation/suite-1500.jsonl');p.add_argument('--output',default='data/evaluation/v2/results')
    p.add_argument('--count',type=int,default=1500);p.add_argument('--nodes',type=int,default=200000)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--stockfish',default='/usr/bin/stockfish')
    p.add_argument('--device',default='cuda');p.add_argument('--seconds',type=float,default=2.0);p.add_argument('--hard-seconds',type=float,default=3.0)
    p.add_argument('--latency-only',action='store_true');p.add_argument('--latency-count',type=int,default=100)
    args=p.parse_args()
    if args.count<500:raise ValueError('Release evaluation requires at least 500 heldout positions')
    rows=suite('data/labels','data/split.json',args.suite,args.count)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    latency_path=output/'latency.json'
    if args.latency_only:
        latency_path.write_text(json.dumps(latency(args,rows),indent=2)+'\n');return
    if not latency_path.exists():raise SystemExit('Run with --latency-only first: it calibrates each model\'s node budget')
    latency_report=json.loads(latency_path.read_text())
    config={name:{'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),'search_nodes':latency_report[name]['nodes_p50']}
            for name,manifest in (('baseline',args.baseline),('candidate',args.candidate))}
    for name,manifest in (('baseline',args.baseline),('candidate',args.candidate)):
        if latency_report[name]['manifest_sha256']!=config[name]['manifest_sha256']:raise SystemExit(f'{name} changed since latency run; rerun --latency-only')
    config.update(suite_sha256=hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),stockfish_nodes=args.nodes,sampling_seed='sha256(game_id:ply)')
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Evaluation configuration changed; use a new output')
    config_path.write_text(json.dumps(config,indent=2))
    moves={}
    for name,manifest in (('baseline',args.baseline),('candidate',args.candidate)):
        path=output/f'{name}.moves.jsonl'
        done={r['key']:r for r in map(json.loads,path.read_text().splitlines())} if path.exists() else {}
        runtime=Runtime(manifest,device=args.device)
        ranks=policy_ranks(runtime,rows);started=time.monotonic()
        with path.open('a') as f:
            for i,(row,rank) in enumerate(zip(rows,ranks)):
                key=f'{row["game_id"]}:{row["ply"]}'
                if key not in done:
                    board=chess.Board(row['fen']);runtime.cache.clear()
                    result=Search(runtime,seconds=math.inf,node_limit=config[name]['search_nodes'],
                                  rng=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16],16))).run(board)
                    done[key]={'key':key,'move':result['move'],'book':result['book'],'rank':rank,'depth':result['depth']}
                    f.write(json.dumps(done[key])+'\n')
                if (i+1)%100==0:f.flush();print(f'{name} search: {i+1}/{len(rows)} ({(i+1)/(time.monotonic()-started):.1f}/s)',flush=True)
        moves[name]=[done[f'{r["game_id"]}:{r["ply"]}'] for r in rows]
        del runtime
    children=[]
    for name in moves:
        for row,move in zip(rows,moves[name]):
            board=chess.Board(row['fen']);board.push_uci(move['move']);children.append(board.fen())
    scores=score_all([r['fen'] for r in rows]+children,output/'stockfish.jsonl',args.stockfish,args.nodes,args.workers)
    results={}
    for name in moves:
        results[name]=[]
        for row,move in zip(rows,moves[name]):
            board=chess.Board(row['fen']);board.push_uci(move['move'])
            loss=max(0,scores[row['fen']]+scores[board.fen()])
            results[name].append({'loss_cp':loss,'blunder':loss>150,'rank':move['rank'],'match':move['move']==row['played_move'],'book':move['book']})
    report=summarize(rows,results,latency_report,config)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('gate','passed')},indent=2),flush=True)
    for name in ('baseline','candidate'):print(name,json.dumps(report[name]['all']),flush=True)
    if not report['passed']:raise SystemExit('Quality gate failed; retain the current release')

if __name__=='__main__':main()
