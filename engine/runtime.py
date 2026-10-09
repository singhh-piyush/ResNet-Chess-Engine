import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import chess
import chess.polyglot
import numpy as np
import torch
from .encoding import board_to_tensor, encode_move
from .model import ChessModel


def book_key(board):
    """Position identity without move counters, matching how the book is built."""
    return ' '.join(board.fen().split()[:4])


def checked_file(manifest_dir, name, sha256, label):
    path = manifest_dir/name
    if path.parent.resolve() != manifest_dir.resolve():
        raise ValueError(f'{label} must be inside manifest directory')
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'{label} checksum mismatch')
    return path


class Runtime:
    def __init__(self, manifest_path='models/release.json', device=None, threads=2, cache_size=8192):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        weights = checked_file(path.parent, manifest['path'], manifest['sha256'], 'Checkpoint')
        self.version = manifest['version']
        self.policy_size = manifest['policy_size']
        if self.policy_size not in (4096, 4272):
            raise ValueError('Unsupported move vocabulary')
        self.format = manifest.get('format', 'torch')
        self.device = device or os.getenv('ENGINE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.format == 'onnx' and self.device == 'cuda' and manifest.get('source_path'):
            # ONNX serves the CPU Space; on a GPU the exported PyTorch source is faster.
            weights = checked_file(path.parent, manifest['source_path'], manifest['source_sha256'], 'Source checkpoint')
            self.format = 'torch'
        if self.format == 'onnx':
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.session = onnxruntime.InferenceSession(str(weights), options, providers=['CPUExecutionProvider'])
            self.device = 'cpu'
        elif self.format == 'torch':
            if self.device == 'cpu':
                torch.set_num_threads(threads)
            self.model = ChessModel(policy_size=self.policy_size, **manifest['architecture'])
            state = torch.load(weights, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state.get('model', state))
            self.model.to(self.device).eval()
        else:
            raise ValueError('Unsupported checkpoint format')
        self.book = {}
        if manifest.get('book'):
            book = checked_file(path.parent, manifest['book'], manifest['book_sha256'], 'Opening book')
            self.book = json.loads(book.read_text())['positions']
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def index(self, move):
        # Old policies cannot distinguish promotions. Prefer queen when using a baseline,
        # but still search all four pieces independently with exact terminal evaluation.
        return encode_move(move) if self.policy_size == 4272 else move.from_square*64+move.to_square

    def infer(self, batch):
        """Raw network outputs for a float32 (N,19,8,8) batch: policy logits and tanh values."""
        if self.format == 'onnx':
            policies, values = self.session.run(None, {'board': batch})
            return policies, values.reshape(-1)
        with torch.inference_mode():
            policies, values = self.model(torch.from_numpy(batch).to(self.device))
        return policies.float().cpu().numpy(), values.float().cpu().numpy().reshape(-1)

    def evaluate(self, boards):
        results = [None]*len(boards)
        missing = []
        for i, board in enumerate(boards):
            # The move-number feature saturates at 60, so it is part of the identity.
            key = (chess.polyglot.zobrist_hash(board), min(board.fullmove_number, 60))
            if key in self.cache:
                self.cache.move_to_end(key)
                results[i] = self.cache[key]
            else:
                missing.append((i, key, board))
        if missing:
            policies, values = self.infer(np.stack([board_to_tensor(b) for _,_,b in missing]))
            for (i, key, _), policy, value in zip(missing, policies, values):
                result = (policy, math.atanh(min(max(float(value), -.999), .999))*400)
                self.cache[key] = result
                results[i] = result
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return results

    def logits(self, board, policy, moves):
        result = np.array([policy[self.index(m)] for m in moves], dtype=np.float64)
        if self.policy_size == 4096:
            # Shared legacy promotion probability mass belongs to the queen by default.
            result -= np.array([8 if m.promotion and m.promotion != chess.QUEEN else 0 for m in moves])
        return result
