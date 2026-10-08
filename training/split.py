"""Freeze game-level splits and preserve the legacy fold-0 holdout."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold


def prepare(labels='data/labels',output='data/split.json',legacy='archive/artifacts/game_ids.npz'):
    path=Path(output)
    old={}
    if Path(legacy).exists():
        groups=np.load(legacy)['arr_0'].astype(str)[::4]
        _,val=next(GroupKFold(n_splits=5).split(np.arange(len(groups)),groups=groups))
        holdout=set(groups[val]); all_old=set(groups)
        for game_id in all_old:
            old[game_id]=('validation' if int(hashlib.sha256(game_id.encode()).hexdigest(),16)%2==0 else 'test') if game_id in holdout else 'train'
    manifest=json.loads(path.read_text()) if path.exists() else {'schema':2,'games':old}
    for source in sorted(Path(labels).glob('*.jsonl')):
        row=json.loads(source.open().readline());game_id=row['game_id']
        if game_id not in manifest['games']:
            bucket=int(hashlib.sha256(game_id.encode()).hexdigest(),16)%10
            manifest['games'][game_id]='train' if bucket<8 else 'validation' if bucket==8 else 'test'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':prepare()
