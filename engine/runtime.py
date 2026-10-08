import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
import chess
import numpy as np
import torch
from .encoding import board_to_tensor, encode_move
from .model import ChessModel


class Runtime:
    def __init__(self, manifest_path='models/release.json'):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        weights = path.parent / manifest['path']
        if weights.parent.resolve() != path.parent.resolve():
            raise ValueError('Checkpoint must be inside manifest directory')
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        self.version = manifest['version']
        self.policy_size = manifest['policy_size']
        if self.policy_size not in (4096, 4272):
            raise ValueError('Unsupported move vocabulary')
        torch.set_num_threads(2)
        self.model = ChessModel(policy_size=self.policy_size, **manifest['architecture'])
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state.get('model', state))
        self.model.eval()
        self.cache = OrderedDict()

    def index(self, move):
        # Old policies cannot distinguish promotions. Prefer queen when using a baseline,
        # but still search all four pieces independently with exact terminal evaluation.
        return encode_move(move) if self.policy_size == 4272 else move.from_square*64+move.to_square

    def evaluate(self, boards):
        results = [None]*len(boards)
        missing = []
        for i, board in enumerate(boards):
            key = (self.version, board.fen())
            if key in self.cache:
                self.cache.move_to_end(key)
                results[i] = self.cache[key]
            else:
                missing.append((i, key, board))
        if missing:
            batch = torch.from_numpy(np.stack([board_to_tensor(b) for _,_,b in missing]))
            with torch.inference_mode():
                policies, values = self.model(batch)
            for (i, key, _), policy, value in zip(missing, policies, values):
                result = (policy.numpy().copy(), math.atanh(float(value.clamp(-.999,.999)))*400)
                self.cache[key] = result
                results[i] = result
                if len(self.cache)>2048:
                    self.cache.popitem(last=False)
        return results

    def logits(self, board, policy, moves):
        result = np.array([policy[self.index(m)] for m in moves], dtype=np.float64)
        if self.policy_size == 4096:
            # Shared legacy promotion probability mass belongs to the queen by default.
            result -= np.array([8 if m.promotion and m.promotion != chess.QUEEN else 0 for m in moves])
        return result
