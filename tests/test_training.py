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


def write_game(labels,game_id,moves,end_time=1_700_000_000):
    import hashlib
    from engine.encoding import encode_move
    board=chess.Board();rows=[]
    for ply,uci in enumerate(moves):
        rows.append({'fen':board.fen(),'policy':encode_move(chess.Move.from_uci(uci)) if ply%2==0 else -1,'value':.1,
                     'game_id':game_id,'ply':ply,'played_move':uci,'metadata':{'end_time':end_time,'time_class':'blitz'}})
        board.push_uci(uci)
    rows.append({'fen':board.fen(),'policy':-1,'value':.1,'game_id':game_id,'ply':len(moves),'metadata':{'end_time':end_time,'time_class':'blitz'}})
    path=labels/(hashlib.sha256(game_id.encode()).hexdigest()+'.jsonl')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return path


def test_packed_inputs_match_board_tensor(tmp_path):
    import numpy as np
    from engine.encoding import board_to_tensor
    from training.pack import pack_game
    from training.train import batch_inputs
    labels=tmp_path/'labels';labels.mkdir()
    path=write_game(labels,'g',['e2e4','d7d5','e4d5','g8f6','f1b5','c7c6'])
    packed=pack_game((str(path),True))
    data={'planes':torch.from_numpy(packed['planes']),'move_number':torch.from_numpy(packed['move_number'])}
    rebuilt=batch_inputs(data,torch.arange(len(packed['policy'])))
    for i,row in enumerate(map(json.loads,path.read_text().splitlines())):
        assert torch.equal(rebuilt[i],torch.from_numpy(board_to_tensor(chess.Board(row['fen']))))
    assert packed['user'].tolist()==[True,False,True,False,True,False,True]
    assert packed['played'][-1]==-1


def test_folds_deterministic_and_exclude_test(tmp_path):
    import numpy as np
    from training.kfold import folds,split
    games=[f'game-{i}' for i in range(40)]
    partition=np.array([2 if i%5==0 else i%2 for i in range(40)],np.uint8).repeat(3)
    data={'meta':{'games':games},'host':{'partition':partition,'game':np.arange(40).repeat(3)},'device':'cpu'}
    first=folds(data,5,tmp_path);second=folds(data,5,tmp_path)
    assert torch.equal(first,second)
    assert (first[torch.from_numpy(partition)==2]==-1).all()
    for fold in range(5):
        train,val=split(first,fold)
        assert len(val) and not set(train.tolist())&set(val.tolist())
        train_games={int(g) for g in data['host']['game'][train.numpy()]};val_games={int(g) for g in data['host']['game'][val.numpy()]}
        assert not train_games&val_games and all(partition[g*3]!=2 for g in train_games|val_games)


def test_book_excludes_test_games_and_search_uses_it(tmp_path):
    import numpy as np
    from training.book import build
    from engine.search import Search
    from tests.test_engine import FakeRuntime
    labels=tmp_path/'labels';labels.mkdir();partitions={}
    for i in range(4):
        write_game(labels,f'train-{i}',['d2d4','d7d5']);partitions[f'train-{i}']='train'
        write_game(labels,f'test-{i}',['e2e4','e7e5']);partitions[f'test-{i}']='test'
    (tmp_path/'split.json').write_text(json.dumps({'games':partitions}))
    book=build(labels,tmp_path/'split.json',minimum=3)['positions']
    assert book=={' '.join(chess.STARTING_FEN.split()[:4]):{'d2d4':4}}
    result=Search(FakeRuntime(),book=book,seconds=.1).run(chess.Board())
    assert result['move']=='d2d4' and result['book']


def test_distillation_pulls_toward_teacher():
    policy=torch.zeros(1,4272,requires_grad=True);value=torch.zeros(1,1)
    losses(policy,value,torch.tensor([-1]),torch.tensor([0.]),soft_index=torch.tensor([[7,9]]),soft_prob=torch.tensor([[.75,.25]]),
           soft_weights=torch.tensor([1.]),alpha=.5).backward()
    assert policy.grad[0,7]<policy.grad[0,9]<0 and policy.grad[0,0]>0
