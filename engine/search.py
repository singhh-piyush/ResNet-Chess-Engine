"""Bounded neural search. Stockfish is never used by the serving process."""
import math
import time
import numpy as np


class SearchStopped(Exception):
    pass


def terminal(board, ply=0):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return (100000-ply) * (1 if outcome.winner == board.turn else -1)


class Search:
    def __init__(self, runtime, seconds=9.5, node_limit=4096, max_depth=4, cancelled=None):
        self.runtime = runtime
        self.seconds = min(seconds, 9.5)
        self.limit = node_limit
        self.max_depth = max_depth
        self.cancelled = cancelled or (lambda:False)
        self.nodes = 0
        self.table = {}

    def check(self):
        if self.cancelled() or time.monotonic() >= self.deadline or self.nodes >= self.limit:
            raise SearchStopped()

    def candidates(self, board, policy, tactical=False):
        moves = list(board.legal_moves)
        logits = self.runtime.logits(board, policy, moves)
        ranked = sorted(zip(moves, logits), key=lambda pair:pair[1], reverse=True)
        if board.is_check():
            return [m for m,_ in ranked]
        if tactical:
            return [m for m,_ in ranked if board.is_capture(m) or m.promotion]
        preferred = {m for m,_ in ranked[:8]}
        return [m for m,_ in ranked if m in preferred or board.is_capture(m) or m.promotion or board.gives_check(m)]

    def negamax(self, board, depth, alpha, beta, ply, quiet=4):
        self.check()
        self.nodes += 1
        value = terminal(board, ply)
        if value is not None:
            return value
        # History belongs to the key: repetition claims cannot share FEN-only scores.
        key = (board.fen(), tuple(m.uci() for m in board.move_stack), depth, quiet)
        if key in self.table:
            return self.table[key]
        original_alpha = alpha
        policy, value = self.runtime.evaluate([board])[0]
        self.check()
        if depth<=0:
            if quiet<=0:
                return value
            moves = self.candidates(board, policy, tactical=True)
            best = -math.inf if board.is_check() else value
            if best>=beta:
                return best
            alpha = max(alpha, best)
        else:
            moves = self.candidates(board, policy)
            best = -math.inf
        cutoff = False
        for move in moves:
            board.push(move)
            try:
                score = -self.negamax(board, depth-1, -beta, -alpha, ply+1, quiet-1 if depth<=0 else quiet)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha>=beta:
                cutoff = True
                break
        # Only store exact results, never an alpha-beta bound.
        if not cutoff and best>original_alpha:
            self.table[key] = best
        return best

    def run(self, board, progress=lambda event:None):
        self.started = time.monotonic()
        self.deadline = self.started+self.seconds
        value = terminal(board)
        if value is not None:
            return {'move':None, 'game_over':True, 'result':board.result(claim_draw=True), 'thinking_log':['Game finished.']}
        legal = list(board.legal_moves)
        # Exact one-ply mate scan includes every root move, independent of policy rank.
        mates = []
        for move in legal:
            self.check()
            board.push(move)
            if board.is_checkmate():
                mates.append(move)
            board.pop()
        policy, _ = self.runtime.evaluate([board])[0]
        logits = self.runtime.logits(board, policy, legal)
        root = mates or self.candidates(board, policy)
        children = []
        for move in root:
            child = board.copy(stack=True)
            child.push(move)
            children.append(child)
        # Batched shallow evaluations provide a legal fallback before deeper iterations.
        evaluated = self.runtime.evaluate(children)
        scores = {m: -(terminal(c,1) if terminal(c,1) is not None else result[1]) for m,c,result in zip(root,children,evaluated)}
        completed = 1
        def report():
            progress({'depth':completed, 'elapsed_ms':round((time.monotonic()-self.started)*1000), 'positions':self.nodes,
                      'message':f'Completed depth {completed}; evaluated {self.nodes} positions.'})
        report()
        if not mates:
            for depth in range(2,self.max_depth+1):
                iteration = {}
                try:
                    for move in sorted(root,key=lambda m:scores[m], reverse=True):
                        board.push(move)
                        try:
                            iteration[move] = -self.negamax(board, depth-1, -math.inf, math.inf, 1)
                        finally:
                            board.pop()
                except SearchStopped:
                    break
                scores = iteration
                completed = depth
                report()
        best = max(scores.values())
        survivors = [m for m in root if best-scores[m]<=100]
        survivor_logits = self.runtime.logits(board, policy, survivors)/.8
        probs = np.exp(survivor_logits-survivor_logits.max())
        probs /= probs.sum()
        chosen = survivors[np.random.default_rng().choice(len(survivors),p=probs)]
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        white_sign = 1 if board.turn else -1
        elapsed = round((time.monotonic()-self.started)*1000)
        return {'move':chosen.uci(), 'confidence':round(float(probabilities[legal.index(chosen)]),4),
                'evaluation':round(scores[chosen]*white_sign/100,2), 'evaluation_source':'neural estimate',
                'candidates':[{'move':m.uci(),'san':board.san(m),'confidence':round(float(probabilities[legal.index(m)]),4),
                    'evaluation':round(scores[m]*white_sign/100,2),
                    'status':'SELECTED' if m==chosen else ('ANALYZED' if m in survivors else 'VETOED')} for m in root],
                'thinking_log':[f'Completed depth {completed}; evaluated {self.nodes} positions in {elapsed} ms.',f'Final decision: {board.san(chosen)}'],
                'is_fallback':False, 'depth':completed, 'positions':self.nodes, 'elapsed_ms':elapsed,
                'model_version':self.runtime.version}
