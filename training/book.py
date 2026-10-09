"""Opening book of your own recent moves, built from train and validation games only."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def build(labels='data/labels', split='data/split.json', months=18, minimum=3):
    partitions=json.loads(Path(split).read_text())['games']
    games=[]
    for source in sorted(Path(labels).glob('*.jsonl')):
        rows=[json.loads(line) for line in source.read_text().splitlines()]
        # The frozen test split never contributes, so evaluation positions stay unseen.
        if rows and partitions[rows[0]['game_id']] in ('train','validation'):games.append(rows)
    latest=max(rows[0]['metadata']['end_time'] for rows in games)
    cutoff=latest-months*30.44*86400
    counts=defaultdict(lambda:defaultdict(int))
    for rows in games:
        if rows[0]['metadata']['end_time']<cutoff:continue
        for row in rows:
            # policy >= 0 marks your moves that lost at most 150 cp.
            if row['policy']>=0:counts[' '.join(row['fen'].split()[:4])][row['played_move']]+=1
    positions={key:{move:n for move,n in sorted(moves.items()) if n>=minimum} for key,moves in sorted(counts.items())}
    return {'version':1,'months':months,'minimum':minimum,'positions':{k:v for k,v in positions.items() if v}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='models/book.json')
    p.add_argument('--months',type=int,default=18);p.add_argument('--minimum',type=int,default=3)
    p.add_argument('--manifest',help='candidate manifest to reference the book from')
    args=p.parse_args()
    book=build(months=args.months,minimum=args.minimum)
    output=Path(args.output);output.write_text(json.dumps(book,indent=1)+'\n')
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    print(f'{len(book["positions"])} book positions, {sum(len(v) for v in book["positions"].values())} moves; sha256 {digest}')
    if args.manifest:
        path=Path(args.manifest)
        if output.parent.resolve()!=path.parent.resolve():raise ValueError('Book must sit beside the manifest')
        manifest=json.loads(path.read_text());manifest.update(book=output.name,book_sha256=digest)
        path.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
