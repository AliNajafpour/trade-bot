"""Reproducible full Chronos-2 adaptation on actual local Wallex crypto data."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.chronos_training import train

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',nargs='+',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--facts-path');parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--context',type=int,default=128);parser.add_argument('--prediction',type=int,default=24)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--eval-limit',type=int,default=48)
    parser.add_argument('--interval-seconds',type=int);parser.add_argument('--eval-steps',type=int,default=100)
    parser.add_argument('--patience',type=int,default=4);parser.add_argument('--status-path')
    args=vars(parser.parse_args());args['paths']=args.pop('data')
    result=train(**args);print(json.dumps({k:result.get(k) for k in ('status','elapsed_seconds','baseline','final','selected_checkpoint')},indent=2))

if __name__=='__main__':main()
