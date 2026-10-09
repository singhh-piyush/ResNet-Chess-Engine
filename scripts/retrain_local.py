"""Run the offline K-fold -> distilled-student pipeline end to end; never auto-promotes weights.

Every stage is resumable, so rerunning after an interruption continues where it stopped.
"""
import json
import subprocess
import sys
from pathlib import Path

STUDENTS = {'10x128':{'num_res_blocks':10, 'channels':128, 'policy_channels':16},
            '8x96':{'num_res_blocks':8, 'channels':96, 'policy_channels':16},
            '6x64':{'num_res_blocks':6, 'channels':64, 'policy_channels':16}}


def stage(name, *command):
    print(f'STAGE {name}', flush=True)
    subprocess.run([sys.executable, '-m', *command], check=True)


def main():
    if not Path('data/packed/meta.json').exists():
        stage('pack labels', 'training.pack')
    stage('k-fold sweep', 'training.kfold', '--stage', 'sweep')
    if not Path('data/kfold/oof.pt').exists():
        stage('out-of-fold teachers', 'training.kfold', '--stage', 'teachers')
    done = {json.loads(line)['name'] for line in Path('data/kfold/students.jsonl').read_text().splitlines()} if Path('data/kfold/students.jsonl').exists() else set()
    for name, architecture in STUDENTS.items():
        if name not in done:
            stage(f'student {name} cross-validation', 'training.kfold', '--stage', 'student', '--cv-fold', '0', '--name', name, '--arch', json.dumps(architecture))
    results = {json.loads(line)['name']:json.loads(line)['metrics'] for line in Path('data/kfold/students.jsonl').read_text().splitlines()}
    best = max(results[n]['style_top1'] for n in STUDENTS)
    # Smallest network within one point of the best style accuracy (dict order is large -> small).
    chosen = [n for n in STUDENTS if results[n]['style_top1'] >= best-.01][-1]
    print(json.dumps({n:{'style_top1':results[n]['style_top1'], 'recent_top1':results[n]['recent_top1']} for n in STUDENTS}), f'chosen {chosen}', flush=True)
    stage(f'student {chosen} final', 'training.kfold', '--stage', 'student', '--final', '--name', chosen, '--arch', json.dumps(STUDENTS[chosen]))
    stage('opening book', 'training.book', '--manifest', f'models/student-{chosen}.json')
    stage('ONNX export', 'scripts.export_onnx', '--manifest', f'models/student-{chosen}.json', '--output', 'models/candidate.json')
    output = f'data/evaluation/v2-student-{chosen}'
    stage('CPU latency calibration', 'evaluation.compare', '--candidate', 'models/candidate.json', '--output', output, '--latency-only')
    stage('independent quality gate', 'evaluation.compare', '--candidate', 'models/candidate.json', '--output', output)
    print('Candidate passed the offline gate. Upload, hosted latency and release promotion still require verification.', flush=True)

if __name__ == '__main__': main()
