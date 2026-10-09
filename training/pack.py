"""Parse every label file once into compact arrays that training loads straight into GPU memory."""
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import chess
import numpy as np
from engine.encoding import board_to_tensor, encode_move

TIME_CLASSES = {'bullet':0, 'blitz':1, 'rapid':2, 'daily':3}
PARTITIONS = {'train':0, 'validation':1, 'test':2}


def user_colors(raw, username):
    colors = {}
    for source in sorted(Path(raw).glob('????-??.json')):
        for game in json.loads(source.read_text())['games']:
            if game.get('url'):
                colors[game['url']] = game['white']['username'].lower() == username.lower()
    return colors


def pack_game(task):
    path, color = task
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    planes = []; move_numbers = []; played = []; user = []
    for row in rows:
        board = chess.Board(row['fen'])
        tensor = board_to_tensor(board)
        planes.append(tensor[:18].astype(np.uint8))
        move_numbers.append(min(board.fullmove_number, 60))
        played.append(encode_move(chess.Move.from_uci(row['played_move'])) if row.get('played_move') else -1)
        user.append(board.turn == color)
    meta = rows[0]['metadata']
    return {'game_id':rows[0]['game_id'], 'planes':np.stack(planes), 'move_number':np.array(move_numbers, np.uint8),
            'policy':np.array([r['policy'] for r in rows], np.int16), 'played':np.array(played, np.int16),
            'user':np.array(user, bool), 'value':np.array([r['value'] for r in rows], np.float32),
            'ply':np.array([r['ply'] for r in rows], np.int16), 'end_time':meta['end_time'],
            'time_class':TIME_CLASSES.get(meta.get('time_class'), 1)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--labels', default='data/labels'); p.add_argument('--split', default='data/split.json')
    p.add_argument('--raw', default='data/raw'); p.add_argument('--username', default='piyushhsingh')
    p.add_argument('--output', default='data/packed'); p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()
    split = json.loads(Path(args.split).read_text())['games']
    colors = user_colors(args.raw, args.username)
    files = sorted(Path(args.labels).glob('*.jsonl'))
    tasks = []
    for path in files:
        game_id = json.loads(path.open().readline())['game_id']
        tasks.append((str(path), colors[game_id]))
    with multiprocessing.Pool(args.workers) as pool:
        games = pool.map(pack_game, tasks, chunksize=16)
    # Game order is fixed by label file name, so indices are reproducible.
    count = [len(g['policy']) for g in games]
    game_index = np.repeat(np.arange(len(games), dtype=np.int32), count)
    arrays = {
        'planes':np.concatenate([g['planes'] for g in games]), 'move_number':np.concatenate([g['move_number'] for g in games]),
        'policy':np.concatenate([g['policy'] for g in games]), 'played':np.concatenate([g['played'] for g in games]),
        'user':np.concatenate([g['user'] for g in games]), 'value':np.concatenate([g['value'] for g in games]),
        'ply':np.concatenate([g['ply'] for g in games]), 'game':game_index,
        'end_time':np.repeat(np.array([g['end_time'] for g in games], np.int64), count),
        'time_class':np.repeat(np.array([g['time_class'] for g in games], np.uint8), count),
        'partition':np.repeat(np.array([PARTITIONS[split[g['game_id']]] for g in games], np.uint8), count)}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output/f'{name}.npy', array)
    dataset = hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()
    meta = {'games':[g['game_id'] for g in games], 'positions':int(len(game_index)), 'dataset_sha256':dataset,
            'split_sha256':hashlib.sha256(Path(args.split).read_bytes()).hexdigest()}
    (output/'meta.json').write_text(json.dumps(meta)+'\n')
    print(f'Packed {len(games)} games, {len(game_index)} positions into {output}', flush=True)

if __name__ == '__main__': main()
