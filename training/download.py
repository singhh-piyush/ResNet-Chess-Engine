"""Refresh every public monthly archive. Interrupted downloads can be resumed."""
import argparse
import json
import time
from pathlib import Path
import requests


def get(session,url):
    for attempt in range(5):
        try:
            r=session.get(url,timeout=45)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt==4:
                raise
            time.sleep(2**attempt)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--username',default='piyushhsingh')
    parser.add_argument('--output',default='data/raw')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers['User-Agent']=f'PersonalChessResearch/2.0 ({args.username})'
    archives=get(session,f'https://api.chess.com/pub/player/{args.username}/games/archives')['archives']
    total=0
    for url in archives:
        month='-'.join(url.rstrip('/').split('/')[-2:])
        path=output/(month+'.json')
        # Refresh current month; historical months are immutable cached inputs.
        import datetime
        if path.exists() and month!=datetime.date.today().strftime('%Y-%m'):
            games=json.loads(path.read_text())['games']
        else:
            response=get(session,url);games=response['games']
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(response));tmp.replace(path)
        total+=len(games)
        print(f'{month}: {len(games)} games',flush=True)
        time.sleep(.2)
    (output/'manifest.json').write_text(json.dumps({'username':args.username,'archives':archives,'games':total},indent=2))
    print(f'Complete: {total} games',flush=True)

if __name__=='__main__':main()
