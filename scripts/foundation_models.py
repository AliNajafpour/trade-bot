"""Download pinned public foundation assets, or verify existing files; no credentials."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def prepare(destination,verify_only=False):
    manifest=json.loads((ROOT/'config/foundation-models.json').read_text('utf-8'))
    verified=[]
    for directory,model in manifest.items():
        for name,expected in model['files'].items():
            target=Path(destination)/directory/name
            if not target.exists() or digest(target)!=expected:
                if verify_only:raise ValueError('Missing or changed model asset: '+str(target))
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.part')
                url=f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}"
                try:
                    with urlopen(url,timeout=120) as response,temporary.open('wb') as output:
                        for block in iter(lambda:response.read(1024*1024),b''):output.write(block)
                    if digest(temporary)!=expected:raise ValueError('Downloaded asset checksum failed: '+str(target))
                    temporary.replace(target)
                finally:temporary.unlink(missing_ok=True)
            verified.append(dict(path=str(target),sha256=expected,bytes=target.stat().st_size))
    return verified


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT/'models')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    files=prepare(args.destination,args.verify_only)
    print(json.dumps(dict(status='ok',files=len(files),bytes=sum(f['bytes'] for f in files))))
