"""Run genuine local Kronos-small token fine-tuning; no live model promotion."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.finetune_kronos import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    for name,default in dict(context=128,prediction=12,stride=16,eval_stride=128,epochs=3,
            batch_size=2,accumulation=8,patience=1,seed=20261021,train_limit=0,eval_limit=0).items():
        parser.add_argument('--'+name.replace('_','-'),type=int,default=default)
    parser.add_argument('--learning-rate',type=float,default=4e-5)
    parser.add_argument('--prediction-hours',type=float,help='Common forecast duration across M5/M15/H1 inputs, up to24h')
    parser.add_argument('--gpu-lock',help='OS lock serializing shared GPU workers')
    parser.add_argument('--smoke',action='store_true',help='Allow <80 days; diagnostic only')
    args = vars(parser.parse_args())
    paths = args.pop('data')
    result = train(paths, **args)
    print(json.dumps({key:result[key] for key in ('status','best_epoch','baseline','final','elapsed_seconds',
        'peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','selected_weights')},indent=2))


if __name__ == '__main__':
    main()
