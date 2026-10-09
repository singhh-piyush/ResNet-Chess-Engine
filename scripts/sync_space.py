"""Sync application files only; preserve versioned model weights and rollback assets."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download

REPO='singhh-piyush/ResNet-Chess-Engine'
ROOT_FILES={'Dockerfile','.dockerignore','README.md','requirements.txt','main.py','.gitattributes'}
APP_DIRS={'engine','backend','chess-frontend','docs'}


def included(path,release_files=frozenset({'models/release.json'})):
    parts=Path(path).parts
    return (path in ROOT_FILES or parts[0] in APP_DIRS or path in release_files) and not any(p in {'node_modules','dist','__pycache__'} for p in parts) and not path.endswith(('.pyc','.log'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='manual');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();api=HfApi()
    manifest=json.loads(Path('models/release.json').read_text())
    weights='models/'+manifest['path']
    # Weight upload is a separate release operation. Never publish a broken reference.
    existing=set(api.list_repo_files(REPO,repo_type='space'))
    if weights not in existing:raise SystemExit(f'Upload and verify {weights} before syncing application code')
    remote=hf_hub_download(REPO,weights,repo_type='space')
    if hashlib.sha256(Path(remote).read_bytes()).hexdigest()!=manifest['sha256']:raise SystemExit('Remote checkpoint checksum mismatch')
    # The opening book is small, versioned in git and checksummed by the manifest.
    release_files={'models/release.json'}|({'models/'+manifest['book']} if manifest.get('book') else set())
    local={str(p) for p in Path('.').rglob('*') if p.is_file() and included(str(p),release_files)}
    # Existing obsolete deployed files are removed from HEAD, recoverable in HF history.
    removals={p for p in existing if not p.startswith('models/') and p not in local}
    print(json.dumps({'revision':args.revision,'files':sorted(local),'remove':sorted(removals),'preserve_models':True},indent=2))
    if args.dry_run:return
    operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=p) for p in sorted(local)]
    operations.extend(CommitOperationDelete(path_in_repo=p) for p in sorted(removals))
    api.create_commit(REPO,repo_type='space',operations=operations,commit_message=f'Deploy GitHub {args.revision}')

if __name__=='__main__':main()
