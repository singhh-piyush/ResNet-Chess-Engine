import json
import math
import chess
import torch
from training.train import losses
from training.split import prepare
from training import mine


def test_policy_mask_keeps_value_learning():
    policy=torch.zeros(2,4272,requires_grad=True)
    value=torch.zeros(2,1,requires_grad=True)
    targets=torch.tensor([12,-1]);evaluations=torch.tensor([.5,-.5])
    losses(policy,value,targets,evaluations).backward()
    assert policy.grad[1].abs().sum()==0
    assert policy.grad[0].abs().sum()>0
    assert value.grad[0].item()<0 and value.grad[1].item()>0


def test_mining_both_turns_and_successor(monkeypatch,tmp_path):
    monkeypatch.setattr(mine,'USERNAME','piyushhsingh')
    monkeypatch.setattr(mine,'score',lambda board:100 if board.turn else -100)
    pgn='[Event "Test"]\n[White "piyushhsingh"]\n[Black "opponent"]\n[Result "*"]\n\n1. e4 e5 *'
    path=tmp_path/'game.jsonl'
    _,count,error=mine.process(('game',pgn,str(path),{}))
    assert error is None and count==3
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['policy']>=0 and rows[1]['policy']==-1 and rows[2]['policy']==-1
    assert rows[0]['value']>0 and rows[1]['value']<0
    assert chess.Board(rows[1]['fen']).turn==chess.BLACK


def test_frozen_game_split(tmp_path):
    labels=tmp_path/'labels';labels.mkdir()
    for i in range(25):(labels/f'{i}.jsonl').write_text(json.dumps({'game_id':f'game-{i}'})+'\n')
    path=tmp_path/'split.json'
    first=prepare(labels,path,legacy=tmp_path/'absent')
    (labels/'new.jsonl').write_text(json.dumps({'game_id':'new-game'})+'\n')
    second=prepare(labels,path,legacy=tmp_path/'absent')
    assert all(second['games'][k]==v for k,v in first['games'].items())
    assert set(second['games'].values())=={'train','validation','test'}


def test_baseline_initialization_preserves_policy_and_promotions():
    from engine.model import ChessModel
    from engine.encoding import PROMOTIONS
    from training.train import initialize_from_baseline
    baseline=ChessModel(num_res_blocks=0,channels=8,policy_size=4096).state_dict()
    candidate=ChessModel(num_res_blocks=0,channels=8)
    initialize_from_baseline(candidate,baseline)
    state=candidate.state_dict()
    assert torch.equal(state['policy_head.5.weight'][:4096],baseline['policy_head.5.weight'])
    for index,move in enumerate(PROMOTIONS,4096):
        old=move.from_square*64+move.to_square
        assert torch.equal(state['policy_head.5.weight'][index],baseline['policy_head.5.weight'][old])
        expected=baseline['policy_head.5.bias'][old]-(0 if move.promotion==chess.QUEEN else 2)
        assert state['policy_head.5.bias'][index]==expected
