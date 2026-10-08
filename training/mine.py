"""Deterministic offline labels with per-game atomic, resumable outputs."""
import argparse
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import chess
import chess.pgn
import chess.engine
from engine.encoding import encode_move

ENGINE=None
USERNAME=None
NODES=None


def init(stockfish,username,nodes):
    global ENGINE,USERNAME,NODES
    ENGINE=chess.engine.SimpleEngine.popen_uci(stockfish)
    ENGINE.configure({'Threads':1,'Hash':64})
    USERNAME=username.lower();NODES=nodes


def score(board):
    outcome=board.outcome(claim_draw=True)
    if outcome:
        return 0 if outcome.winner is None else (10000 if outcome.winner==board.turn else -10000)
    # Clear hash for reproducible fixed-node labels independent of worker scheduling.
    ENGINE.configure({'Clear Hash':None})
    return ENGINE.analyse(board,chess.engine.Limit(nodes=NODES))['score'].pov(board.turn).score(mate_score=10000)


def process(task):
    game_id,pgn,path,metadata=task
    try:
        game=chess.pgn.read_game(io.StringIO(pgn))
        if game is None or game.errors:raise ValueError('Malformed PGN')
        board=game.board()
        if not board.is_valid():raise ValueError('Invalid initial board')
        our_color=True if game.headers.get('White','').lower()==USERNAME else False
        if game.headers.get('White','').lower()!=USERNAME and game.headers.get('Black','').lower()!=USERNAME:
            raise ValueError('Game does not contain requested player')
        moves=list(game.mainline_moves())
        positions=[]; values=[]
        for move in moves:
            if move not in board.legal_moves:raise ValueError('Illegal PGN move')
            positions.append((board.fen(),board.turn,encode_move(move)))
            values.append(score(board));board.push(move)
        final_score=score(board)
        rows=[]
        for ply,(fen,turn,target) in enumerate(positions):
            following=values[ply+1] if ply+1<len(values) else final_score
            cp_loss=values[ply]+following
            policy_target=target if turn==our_color and cp_loss<=150 else -1
            rows.append({'fen':fen,'policy':policy_target,'value':math.tanh(values[ply]/400),
                         'game_id':game_id,'ply':ply,'played_move':moves[ply].uci(),'metadata':metadata})
        rows.append({'fen':board.fen(),'policy':-1,'value':math.tanh(final_score/400),'game_id':game_id,'ply':len(moves),'metadata':metadata})
        dest=Path(path);tmp=dest.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(row)+'\n' for row in rows));tmp.replace(dest)
        return game_id,len(rows),None
    except Exception as exc:
        return game_id,0,str(exc)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='data/raw');p.add_argument('--output',default='data/labels')
    p.add_argument('--username',default='piyushhsingh');p.add_argument('--stockfish',default='/usr/bin/stockfish')
    p.add_argument('--nodes',type=int,default=20000);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config={'username':args.username,'nodes':args.nodes,'stockfish_sha256':hashlib.sha256(Path(args.stockfish).read_bytes()).hexdigest(),'encoding':2,'augmentation':'none'}
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Label settings changed; choose a new output directory')
    config_path.write_text(json.dumps(config,indent=2))
    tasks=[];seen=set();cached=0
    for source in sorted(Path(args.raw).glob('????-??.json')):
        for game in json.loads(source.read_text())['games']:
            if game.get('rules')!='chess':continue
            game_id=game.get('url') or hashlib.sha256(game['pgn'].encode()).hexdigest()
            if game_id in seen:continue
            seen.add(game_id);dest=output/(hashlib.sha256(game_id.encode()).hexdigest()+'.jsonl')
            if dest.exists():cached+=1;continue
            metadata={k:game.get(k) for k in ('time_class','time_control','rated','end_time')}
            metadata['source']=source.name;metadata['pgn_sha256']=hashlib.sha256(game['pgn'].encode()).hexdigest()
            tasks.append((game_id,game['pgn'],str(dest),metadata))
    print(f'{len(seen)} unique standard games; {cached} cached; {len(tasks)} remaining',flush=True)
    errors=[];positions=0
    with multiprocessing.Pool(max(1,min(args.workers,8)),initializer=init,initargs=(args.stockfish,args.username,args.nodes)) as pool:
        for i,(game_id,count,error) in enumerate(pool.imap_unordered(process,tasks,chunksize=1),1):
            positions+=count
            if error:errors.append({'game_id':game_id,'error':error})
            if i%10==0 or error:print(f'{i}/{len(tasks)} games; {positions} positions; {len(errors)} errors',flush=True)
    (output/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise SystemExit(f'{len(errors)} games failed; inspect errors.json and rerun')
    print('Labels complete',flush=True)

if __name__=='__main__':main()
