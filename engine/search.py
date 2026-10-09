"""Bounded neural search. Stockfish is never used by the serving process."""
import math
import time
import chess
import chess.polyglot
import numpy as np
from .runtime import book_key

MATE = 100000
# Root moves within this many centipawns of the best survive the blunder veto.
MARGIN = 100
EXACT, LOWER, UPPER = 0, 1, 2
PIECE_VALUES = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}


class SearchStopped(Exception):
    pass


def terminal(board, ply=0):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return (MATE-ply) * (1 if outcome.winner == board.turn else -1)


def fast_terminal(board, ply):
    """In-search terminal check without replaying history for threefold claims:
    any repetition inside the search is scored as a draw."""
    if not any(board.generate_legal_moves()):
        return -(MATE-ply) if board.is_check() else 0.0
    if board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(2):
        return 0.0
    return None


def history_chain(board):
    """Transposition key covering every position since the last irreversible move, so
    positions reached through different repetition histories never share a score."""
    replay = board.copy(stack=True)
    hashes = [chess.polyglot.zobrist_hash(replay)]
    for _ in range(min(board.halfmove_clock, len(replay.move_stack))):
        replay.pop()
        hashes.append(chess.polyglot.zobrist_hash(replay))
    return hash(tuple(hashes))


def child_chain(board, chain):
    """Key of `board` after a move was pushed onto the position keyed by `chain`."""
    position = chess.polyglot.zobrist_hash(board)
    return position if board.halfmove_clock == 0 else hash((chain, position))


def mvv_lva(board, move):
    victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    gain = (PIECE_VALUES[victim] if victim else 0) + (PIECE_VALUES[move.promotion] if move.promotion else 0)
    return gain*10 - PIECE_VALUES[board.piece_type_at(move.from_square)]


class Search:
    def __init__(self, runtime, seconds=2.0, hard_seconds=None, node_limit=200000, max_depth=8, quiet=3,
                 cancelled=None, rng=None, book=None, batch_horizon=True):
        """`seconds` is the soft budget: no iteration starts that is unlikely to finish
        before it. `hard_seconds` (default 1.5x) aborts an iteration in progress."""
        self.runtime = runtime
        self.rng = rng if rng is not None else np.random.default_rng()
        self.soft = seconds
        self.hard = hard_seconds if hard_seconds is not None else seconds*1.5
        self.limit = node_limit
        self.max_depth = max_depth
        self.quiet = quiet
        self.cancelled = cancelled or (lambda:False)
        self.book = getattr(runtime, 'book', {}) if book is None else book
        # One batched call for all horizon children beats sequential calls on CPU and GPU,
        # even counting children a beta cutoff would have skipped.
        self.batch_horizon = batch_horizon
        self.nodes = 0
        self.table = {}

    def check(self):
        if self.cancelled() or self.nodes >= self.limit or time.monotonic() >= self.deadline:
            raise SearchStopped()

    def candidates(self, board, policy, tactical=False, recapture=None, first=None):
        moves = list(board.legal_moves)
        if tactical and not board.is_check():
            moves = [m for m in moves if (board.is_capture(m) or m.promotion) and (recapture is None or m.to_square == recapture)]
            return sorted(moves, key=lambda m:mvv_lva(board, m), reverse=True)
        logits = self.runtime.logits(board, policy, moves)
        ranked = [m for m,_ in sorted(zip(moves, logits), key=lambda pair:pair[1], reverse=True)]
        if not board.is_check():
            preferred = set(ranked[:8])
            ranked = [m for m in ranked if m in preferred or board.is_capture(m) or m.promotion or board.gives_check(m)]
        if first in ranked:
            ranked.remove(first)
            ranked.insert(0, first)
        return ranked

    def negamax(self, board, depth, alpha, beta, ply, quiet, chain=None):
        self.check()
        self.nodes += 1
        value = fast_terminal(board, ply)
        if value is not None:
            return value
        first = None
        if depth > 0:
            entry = self.table.get(chain)
            if entry is not None:
                stored_depth, score, flag, first = entry
                if stored_depth >= depth and (flag == EXACT or (flag == LOWER and score >= beta) or (flag == UPPER and score <= alpha)):
                    return score
        original_alpha = alpha
        policy, value = self.runtime.evaluate([board])[0]
        self.check()
        if depth <= 0:
            if quiet <= 0:
                return value
            # Deep in quiescence only recaptures on the last destination square remain.
            recapture = board.peek().to_square if self.quiet-quiet >= 2 and board.move_stack else None
            moves = self.candidates(board, policy, tactical=True, recapture=recapture)
            best = -math.inf if board.is_check() else value
            if best >= beta:
                return best
            alpha = max(alpha, best)
        else:
            moves = self.candidates(board, policy, first=first)
            best = -math.inf
            if depth == 1 and self.batch_horizon:
                children = []
                for move in moves:
                    child = board.copy(stack=False)
                    child.push(move)
                    children.append(child)
                self.runtime.evaluate(children)
        best_move = None
        for move in moves:
            board.push(move)
            try:
                score = -self.negamax(board, depth-1, -beta, -alpha, ply+1, quiet-1 if depth <= 0 else quiet,
                                      child_chain(board, chain) if depth > 1 else None)
            finally:
                board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        if depth > 0:
            flag = LOWER if best >= beta else UPPER if best <= original_alpha else EXACT
            self.table[chain] = (depth, best, flag, best_move)
        return best

    def book_result(self, board, moves, policy, probabilities, legal):
        counts = np.array([self.book[book_key(board)][m.uci()] for m in moves], dtype=np.float64)
        chosen = moves[self.rng.choice(len(moves), p=counts/counts.sum())]
        children = []
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            children.append(child)
        white_sign = 1 if board.turn else -1
        scores = {m: -result[1] for m, result in zip(moves, self.runtime.evaluate(children))}
        elapsed = round((time.monotonic()-self.started)*1000)
        played = int(counts[moves.index(chosen)])
        return {'move':chosen.uci(), 'confidence':round(float(probabilities[legal.index(chosen)]),4),
                'evaluation':round(scores[chosen]*white_sign/100,2), 'evaluation_source':'neural estimate',
                'candidates':[{'move':m.uci(),'san':board.san(m),'confidence':round(float(probabilities[legal.index(m)]),4),
                    'evaluation':round(scores[m]*white_sign/100,2),'status':'SELECTED' if m==chosen else 'ANALYZED'} for m in moves],
                'thinking_log':[f'Opening book: played {played} time{"s" if played != 1 else ""} from this position.',
                                f'Final decision: {board.san(chosen)}'],
                'is_fallback':False, 'book':True, 'depth':0, 'positions':len(children)+1, 'elapsed_ms':elapsed,
                'model_version':self.runtime.version}

    def run(self, board, progress=lambda event:None):
        self.started = time.monotonic()
        self.deadline = self.started+self.hard
        value = terminal(board)
        if value is not None:
            return {'move':None, 'game_over':True, 'result':board.result(claim_draw=True), 'thinking_log':['Game finished.']}
        legal = list(board.legal_moves)
        # Exact one-ply mate scan includes every root move, independent of policy rank.
        mates = []
        for move in legal:
            board.push(move)
            if board.is_checkmate():
                mates.append(move)
            board.pop()
        policy, _ = self.runtime.evaluate([board])[0]
        logits = self.runtime.logits(board, policy, legal)
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        if self.book and not mates:
            entry = self.book.get(book_key(board), {})
            booked = [m for m in legal if m.uci() in entry]
            if booked:
                return self.book_result(board, booked, policy, probabilities, legal)
        root = mates or self.candidates(board, policy)
        children = []
        for move in root:
            child = board.copy(stack=True)
            child.push(move)
            children.append(child)
        # Batched shallow evaluations provide a legal fallback before deeper iterations.
        evaluated = self.runtime.evaluate(children)
        self.nodes = 1 + len(children)
        scores = {}
        for move, child, result in zip(root, children, evaluated):
            exact = terminal(child, 1)
            scores[move] = -(exact if exact is not None else result[1])
        completed = 1
        def report():
            progress({'depth':completed, 'elapsed_ms':round((time.monotonic()-self.started)*1000), 'positions':self.nodes,
                      'preview_moves':[m.uci() for m in sorted(scores,key=scores.get,reverse=True)[:3]],
                      'message':f'Completed depth {completed}; evaluated {self.nodes} positions.'})
        report()
        favourite = legal[int(probabilities.argmax())]
        last_duration, growth = time.monotonic()-self.started, 3
        root_chain = history_chain(board)
        if not mates and len(root) > 1:
            for depth in range(2, self.max_depth+1):
                started = time.monotonic()
                # Never start an iteration the soft budget cannot fit; branching makes each
                # iteration take several times longer than the previous one.
                if started-self.started+last_duration*growth > self.soft:
                    break
                # A confident policy move that survives a two-ply check needs no deeper look.
                if completed >= 2 and probabilities.max() > .85 and max(scores, key=scores.get) == favourite:
                    break
                iteration = {}
                best = -math.inf
                try:
                    for move in sorted(root, key=lambda m:scores[m], reverse=True):
                        progress({'depth':depth, 'positions':self.nodes, 'elapsed_ms':round((time.monotonic()-self.started)*1000),
                                  'preview_moves':[move.uci()], 'message':f'Considering {board.san(move)} at depth {depth}.'})
                        # Only scores within MARGIN of the best matter; anything that fails
                        # below that window is vetoed exactly as an exact score would be.
                        alpha = -math.inf if best == -math.inf else best-MARGIN-1
                        board.push(move)
                        try:
                            score = -self.negamax(board, depth-1, -math.inf, -alpha, 1, self.quiet, child_chain(board, root_chain))
                        finally:
                            board.pop()
                        iteration[move] = score
                        best = max(best, score)
                except SearchStopped:
                    break
                scores = iteration
                completed = depth
                duration = time.monotonic()-started
                if depth > 2:
                    growth = max(3, duration/max(last_duration, 1e-3))
                last_duration = duration
                report()
        best = max(scores.values())
        survivors = [m for m in root if best-scores[m]<=MARGIN]
        survivor_logits = self.runtime.logits(board, policy, survivors)/.8
        weights = np.exp(survivor_logits-survivor_logits.max())
        weights /= weights.sum()
        chosen = survivors[self.rng.choice(len(survivors),p=weights)]
        white_sign = 1 if board.turn else -1
        elapsed = round((time.monotonic()-self.started)*1000)
        return {'move':chosen.uci(), 'confidence':round(float(probabilities[legal.index(chosen)]),4),
                'evaluation':round(scores[chosen]*white_sign/100,2), 'evaluation_source':'neural estimate',
                'candidates':[{'move':m.uci(),'san':board.san(m),'confidence':round(float(probabilities[legal.index(m)]),4),
                    'evaluation':round(scores[m]*white_sign/100,2),
                    'status':'SELECTED' if m==chosen else ('ANALYZED' if m in survivors else 'VETOED')} for m in root],
                'thinking_log':[f'Completed depth {completed}; evaluated {self.nodes} positions in {elapsed} ms.',f'Final decision: {board.san(chosen)}'],
                'is_fallback':False, 'book':False, 'depth':completed, 'positions':self.nodes, 'elapsed_ms':elapsed,
                'model_version':self.runtime.version}
