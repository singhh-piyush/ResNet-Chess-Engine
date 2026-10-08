"""Continue the authorized offline run after labels finish; never auto-promote weights."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def stage(name,command):
    print(f'STAGE {name}',flush=True)
    subprocess.run([sys.executable,'-m',*command],check=True)


def main():
    log=Path('data/mine.log')
    print('Waiting for the active relabeling run to finish',flush=True)
    while True:
        text=log.read_text()
        if 'Traceback' in text or 'games failed' in text:raise SystemExit('Relabeling failed; training stopped')
        if 'Labels complete' in text:break
        time.sleep(10)
    errors=json.loads(Path('data/labels/errors.json').read_text())
    if errors:raise SystemExit('Label errors require repair')
    stage('freeze splits',['training.split'])
    stage('GPU training',['training.train','--batch','256'])
    source=Path('data/run/best.pth')
    dest=Path('models/candidate-v2.pth');dest.write_bytes(source.read_bytes())
    manifest={'path':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'policy_size':4272,
              'version':'candidate-v2-'+hashlib.sha256(dest.read_bytes()).hexdigest()[:12],
              'architecture':{'num_res_blocks':15,'channels':192},'quality_status':'pending independent gate'}
    Path('models/candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    stage('500-position independent quality gate',['evaluation.compare','--candidate','models/candidate.json'])
    print('Candidate passed offline quality gate. Upload, hosted latency and release promotion still require verification.',flush=True)

if __name__=='__main__':main()
