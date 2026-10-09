"""Measure search latency and depth on frozen suite positions, as the CPU Space would see them."""
import argparse
import json
import time
from pathlib import Path
import chess
import numpy as np
from engine.runtime import Runtime
from engine.search import Search


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='models/release.json');p.add_argument('--suite',default='data/evaluation/suite.jsonl')
    p.add_argument('--count',type=int,default=50);p.add_argument('--seconds',type=float,default=2.0)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    runtime=Runtime(args.manifest,device=args.device,threads=args.threads)
    rows=[json.loads(line) for line in Path(args.suite).read_text().splitlines()][:args.count]
    runtime.evaluate([chess.Board()])
    latency=[];depth=[];nodes=[]
    for i,row in enumerate(rows):
        board=chess.Board(row['fen']);start=time.monotonic()
        result=Search(runtime,seconds=args.seconds,rng=np.random.default_rng(i)).run(board)
        latency.append(time.monotonic()-start);depth.append(result['depth']);nodes.append(result['positions'])
    report={'positions':len(rows),'seconds_budget':args.seconds,'latency_p50':float(np.percentile(latency,50)),
            'latency_p95':float(np.percentile(latency,95)),'latency_max':max(latency),'depth_mean':float(np.mean(depth)),
            'depth_counts':{str(d):depth.count(d) for d in sorted(set(depth))},'nodes_median':float(np.median(nodes))}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
