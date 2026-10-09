"""Export a verified PyTorch release to ONNX for faster CPU serving; never changes the source."""
import argparse
import hashlib
import json
from pathlib import Path
import chess
import numpy as np
import torch
from engine.encoding import board_to_tensor
from engine.runtime import Runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    source=Path(args.manifest);manifest=json.loads(source.read_text())
    runtime=Runtime(source,device='cpu')
    model=runtime.model.float().eval()
    weights=Path(args.output).with_suffix('.onnx')
    if weights.parent.resolve()!=source.parent.resolve():raise ValueError('Export beside the source manifest')
    example=torch.from_numpy(np.stack([board_to_tensor(chess.Board())]*2))
    # ONNX Runtime fuses BatchNorm into the convolutions during graph optimization.
    torch.onnx.export(model,(example,),str(weights),input_names=['board'],output_names=['policy','value'],
                      dynamic_shapes={'x':{0:torch.export.Dim('batch',min=1,max=4096)}},dynamo=True,external_data=False)
    exported=dict(manifest,path=weights.name,sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),format='onnx',
                  source_path=manifest['path'],source_sha256=manifest['sha256'])
    Path(args.output).write_text(json.dumps(exported,indent=2)+'\n')
    onnx=Runtime(args.output,device='cpu')
    boards=[chess.Board()]
    for uci in ('e2e4','c7c5','g1f3','d7d6','d2d4','c5d4'):boards.append(boards[-1].copy());boards[-1].push_uci(uci)
    batch=np.stack([board_to_tensor(b) for b in boards])
    (tp,tv),(op,ov)=runtime.infer(batch),onnx.infer(batch)
    error=max(float(np.abs(tp-op).max()),float(np.abs(tv-ov).max()))
    if error>1e-3:raise ValueError(f'ONNX outputs differ from PyTorch by {error}')
    print(f'Exported {weights} (max abs difference {error:.2e})')

if __name__=='__main__':main()
