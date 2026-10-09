import json
import math
import time
import chess
import numpy as np
import pytest
from fastapi.testclient import TestClient
from engine.encoding import PROMOTIONS, POLICY_SIZE, encode_move, decode_move, board_to_tensor
from engine.search import Search, terminal
from engine.runtime import Runtime
from backend import app as backend


class FakeRuntime:
    version='test'
    def index(self,move):return encode_move(move)
    def logits(self,board,policy,moves):return np.array([policy[encode_move(m)] for m in moves])
    def evaluate(self,boards):return [(np.zeros(POLICY_SIZE),0.) for board in boards]


def test_promotions_unique():
    assert len(PROMOTIONS)==176
    assert len(set(encode_move(m) for m in PROMOTIONS))==176
    for move in PROMOTIONS:assert decode_move(encode_move(move))==move
    for fen in ('7k/P7/8/8/8/8/8/7K w - - 0 1','7k/8/8/8/8/8/p7/7K b - - 0 1'):
        board=chess.Board(fen)
        moves=[m for m in board.legal_moves if m.promotion]
        assert len({encode_move(m) for m in moves})==4


def test_castling_ep_and_features():
    board=chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert all(board_to_tensor(board)[i].all() for i in range(13,17))
    assert chess.Move.from_uci('e1g1') in board.legal_moves
    board=chess.Board('7k/8/8/3pP3/8/8/8/7K w - d6 0 1')
    assert chess.Move.from_uci('e5d6') in board.legal_moves
    assert board_to_tensor(board)[17,5,3]==1


def test_terminal_mate_and_stalemate():
    assert terminal(chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'))==-100000
    assert terminal(chess.Board('7k/5Q2/5K2/8/8/8/8/8 b - - 0 1'))==0


def test_mate_found_outside_policy():
    board=chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    result=Search(FakeRuntime(),seconds=.5).run(board)
    board.push_uci(result['move']);assert board.is_checkmate()


def test_deadline_and_legal_result():
    board=chess.Board();start=time.monotonic()
    result=Search(FakeRuntime(),seconds=.05).run(board)
    assert chess.Move.from_uci(result['move']) in board.legal_moves
    assert time.monotonic()-start<.3
    assert board.fen()==chess.STARTING_FEN


def test_api_validation_stream_and_busy(monkeypatch):
    monkeypatch.setattr(backend,'runtime',FakeRuntime())
    client=TestClient(backend.app)
    assert client.post('/predict',json={'fen':'garbage'}).status_code==422
    assert client.post('/predict',json={'fen':chess.STARTING_FEN,'moves':['e2e4']}).status_code==422
    board=chess.Board()
    history=['g1f3','g8f6','f3g1','f6g8']*2
    for move in history:board.push_uci(move)
    response=client.post('/predict/stream',json={'fen':board.fen(),'moves':history})
    assert 'event: result' in response.text and '"game_over": true' in response.text
    backend.busy.acquire()
    try:assert client.post('/predict',json={'fen':chess.STARTING_FEN}).status_code==429
    finally:backend.busy.release()
    monkeypatch.setattr(backend,'runtime',None)
    assert client.get('/ready').status_code==503
    assert client.post('/offer_draw',json={'fen':chess.STARTING_FEN}).status_code==503


def test_checksum_rejected(tmp_path):
    weights=tmp_path/'bad.pth';weights.write_bytes(b'bad')
    path=tmp_path/'release.json';path.write_text(json.dumps({'path':'bad.pth','sha256':'incorrect'}))
    with pytest.raises(ValueError,match='checksum'):Runtime(path)


def test_baseline_queen_bias():
    runtime=Runtime()
    board=chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
    moves=[m for m in board.legal_moves if m.promotion]
    if runtime.policy_size != 4096:
        assert len({runtime.index(m) for m in moves}) == 4
        return
    logits=runtime.logits(board,np.zeros(4096),moves)
    assert moves[int(logits.argmax())].promotion==chess.QUEEN


def test_live_previews_are_legal_and_preserve_position():
    board=chess.Board(); original=board.fen(); events=[]
    result=Search(FakeRuntime(),seconds=.05).run(board,events.append)
    assert events
    assert any('Considering ' in event['message'] for event in events)
    for event in events:
        assert 1 <= len(event['preview_moves']) <= 3
        assert all(chess.Move.from_uci(move) in board.legal_moves for move in event['preview_moves'])
    assert board.fen()==original
    assert chess.Move.from_uci(result['move']) in board.legal_moves


class MaterialRuntime(FakeRuntime):
    """Deterministic evaluator: material balance from the side to move, policy prefers captures."""
    values={chess.PAWN:100,chess.KNIGHT:300,chess.BISHOP:310,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
    def evaluate(self,boards):
        results=[]
        for board in boards:
            score=sum(self.values[p.piece_type]*(1 if p.color==board.turn else -1) for p in board.piece_map().values())
            policy=np.zeros(POLICY_SIZE)
            for move in board.legal_moves:policy[encode_move(move)]=(move.from_square*7+move.to_square)%11
            results.append((policy,float(score)))
        return results


class NoTable(dict):
    def __setitem__(self,key,value):pass


TACTICAL=('r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7',
          'r2qk2r/ppp2ppp/2n1bn2/2bpp3/4P3/2PP1N2/PP1NBPPP/R1BQK2R w KQkq - 0 7',
          'r3k2r/pp1n1ppp/2p1pn2/q2p4/2PP4/2N1PN2/PPQ2PPP/R3KB1R w KQkq - 0 10')


@pytest.mark.parametrize('fen',TACTICAL)
def test_windowed_tt_search_matches_full_window(fen,monkeypatch):
    from engine import search
    def scores(full):
        if full:monkeypatch.setattr(search,'MARGIN',10**9)
        else:monkeypatch.setattr(search,'MARGIN',100)
        engine=Search(MaterialRuntime(),seconds=math.inf,max_depth=3,rng=np.random.default_rng(0))
        if full:engine.table=NoTable()
        result=engine.run(chess.Board(fen))
        assert result['depth']==3
        return {c['move']:(c['evaluation'],c['status']) for c in result['candidates']}
    reference=scores(True);windowed=scores(False)
    best=max(e for e,_ in reference.values())
    for move,(evaluation,_) in reference.items():
        if best-evaluation<=1:
            # Survivors keep their exact score; everything else must be vetoed.
            assert windowed[move][0]==evaluation
        else:
            assert windowed[move][1]=='VETOED'


def test_hard_cap_bounds_slow_evaluator():
    class Slow(FakeRuntime):
        def evaluate(self,boards):
            time.sleep(.01);return super().evaluate(boards)
    start=time.monotonic()
    result=Search(Slow(),seconds=.2,hard_seconds=.3).run(chess.Board())
    assert time.monotonic()-start<.5
    assert chess.Move.from_uci(result['move']) in chess.Board().legal_moves


def test_single_legal_move_returns_immediately():
    board=chess.Board('7k/8/8/8/8/8/6q1/7K w - - 0 1')
    result=Search(MaterialRuntime(),seconds=5).run(board)
    assert result['move']=='h1g2' and result['depth']==1
