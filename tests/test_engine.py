import json
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
