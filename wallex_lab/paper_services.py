"""Lightweight local paper-service recovery; no exchange order transport or model imports."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from .locking import exclusive, atomic_replace

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'runs'/'paper-services'


def resources():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
        value=Memory(); value.length=ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError('GlobalMemoryStatusEx failed')
        available=value.available; total=value.total
        drives={REPO.anchor, os.environ.get('SystemDrive','C:')+'\\'}
    else:
        info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
        available=info['MemAvailable'];total=info['MemTotal'];drives={REPO.anchor}
    return dict(available_mb=round(available/1024**2),total_mb=round(total/1024**2),
                disks_free_mb=[round(shutil.disk_usage(drive).free/1024**2) for drive in sorted(drives)])


def admission(snapshot, minimum_mb):
    if snapshot['available_mb'] < minimum_mb: return 'low_available_memory'
    if min(snapshot['disks_free_mb']) < 2048: return 'low_disk_space'
    return None


@contextmanager
def heavy_slot(root=ROOT):
    """One heavy model load across paper producers; never holds a protection lock."""
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'heavy-model.lock'):
        blocked=admission(resources(),4096)
        if blocked: raise RuntimeError('Heavy inference deferred: '+blocked)
        yield


def amended_manifest(manifest, identity, reason):
    changed={key for key in manifest['identity'].keys()|identity.keys() if manifest['identity'].get(key)!=identity.get(key)}
    if not changed <= {'runner_sha256','source_sha256','transport_sha256','policy_sha256'}:
        raise ValueError('Recovery cannot change capital, fees/configuration, window, universe or model identity')
    result=deepcopy(manifest)
    result.setdefault('identity_amendments',[]).append(dict(applied_at=datetime.now(timezone.utc).isoformat(),
        reason=reason,previous_identity=manifest['identity'],new_identity=identity,
        comparison_limitation='Code changed at recovery; analyze pre/post periods separately. Accounts and original entry deadline retained.'))
    result['identity']=identity
    return result


def retry_delay(failures):
    return min(300,15*2**min(max(0,failures-1),5))


def health(spec, stamp=None):
    try:
        path=Path(spec['heartbeat'])
        with path.open(encoding='utf-8') as handle: prefix=handle.read(4096)
        first=re.match(r'\s*\{\s*"updated_at"\s*:\s*"([^"]+)"',prefix)
        if first: updated=first[1]
        else:
            if path.stat().st_size>16*1024**2: return False
            updated=json.loads(path.read_text(encoding='utf-8'))['updated_at']
        age=((stamp or datetime.now(timezone.utc))-datetime.fromisoformat(updated.replace('Z','+00:00'))).total_seconds()
        if not 0<=age<120: return False
        with urlopen('http://127.0.0.1:'+str(spec['port'])+'/',timeout=2) as response:
            response.read(128*1024)
            return response.status==200
    except (OSError,ValueError,KeyError,TypeError): return False


def services():
    portable=sys.executable
    torch=sys.executable
    seed=str(REPO/'runs/account-sized-wallex-20261023/capital-seed.json')
    source=str(REPO/'runs/account-sized-wallex-20261023/source')
    specs=[]
    for name,config,root,port in (
        ('shared-source','runs/account-sized-wallex-20261023/source-config.json',source,8789),
        ('original-council','config/market-council.example.json',str(REPO/'runs/market-council-20261022'),8784)):
        specs.append(dict(name=name,argv=[portable,'-B','-m','wallex_lab.market_council','--config',str(REPO/config)],
                          heartbeat=str(Path(root)/'runtime-state.json'),port=port,minimum_mb=3072))
    for name,folder,mode,ports in (
        ('adaptive-comparison','adaptive-wallex-20261023','adaptive',(8830,8831,8832)),
        ('strategy-comparison','strategy-bank-20261023','strategies',(8840,8841,8842))):
        root=REPO/'runs'/folder/'comparison'
        argv=[portable,'-B',str(REPO/'scripts/compare_council_policies.py'),'--config',str(REPO/'runs'/folder/'config.json'),
              '--root',str(root),'--source',source,'--capital-seed',seed,'--policy-mode',mode,'--seconds','7200',
              '--old-port',str(ports[0]),'--fixed-port',str(ports[1]),'--port',str(ports[2]),'--resume-code-update']
        specs.append(dict(name=name,argv=argv,heartbeat=str(root/'comparison-state.json'),port=ports[2],minimum_mb=3072,
                          comparison=str(root/'comparison-state.json')))
    for name,folder,module,port in (
        ('weekly-news','survival-weekly-news-20261023','wallex_lab.survival_news',8850),
        ('survival-reference','survival-paper-account-sized-20261023','wallex_lab.survival_paper',8792)):
        root=REPO/'runs'/folder
        specs.append(dict(name=name,argv=[torch,'-B','-m',module,'--config',str(root/'config.json')],
                          heartbeat=str(root/'status.json'),port=port,minimum_mb=4096))
    return specs


def write(path, value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
    atomic_replace(temporary,path)


def event(name, **fields):
    path=ROOT/'events.jsonl'
    # Bounded operational logs only; experiment data and account ledgers are never pruned.
    if path.exists() and path.stat().st_size>5*1024**2:
        for suffix in (3,2,1):
            previous=ROOT/f'events.{suffix}.jsonl'
            if previous.exists():
                if suffix==3: previous.unlink()
                else: atomic_replace(previous,ROOT/f'events.{suffix+1}.jsonl')
        atomic_replace(path,ROOT/'events.1.jsonl')
    with path.open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),event=name,**fields))+'\n')


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.3): return True
    except OSError: return False


def stop_owned(process):
    if process.poll() is not None: return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        process.terminate(); process.wait(timeout=15)


def run(locked=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    with nullcontext() if locked else exclusive(ROOT/'supervisor.lock'):
        items=[dict(spec=spec,process=None,failures=0,next_start=0.,started=0.,state='pending') for spec in services()]
        event('supervisor_started',pid=os.getpid())
        while not (ROOT/'stop-request.json').exists():
            mono=time.monotonic(); stamp=datetime.now(timezone.utc); snapshot=resources()
            started_one=False
            for item in items:
                spec=item['spec']; process=item['process']; running=process and process.poll() is None
                if health(spec):
                    item.update(state='running' if running else 'adopted_running',failures=0)
                    continue
                if spec.get('comparison') and Path(spec['comparison']).exists():
                    if json.loads(Path(spec['comparison']).read_text()).get('status')=='complete_flat':
                        item['state']='completed'; continue
                if running:
                    if mono-item['started']<180:
                        item['state']='starting'; continue
                    stop_owned(process); event('stale_service_restarted',service=spec['name'],pid=process.pid)
                if process:
                    item['failures']+=1
                    item.update(process=None,next_start=mono+retry_delay(item['failures']),state='restart_backoff')
                    event('service_exit',service=spec['name'],returncode=process.poll(),retry_seconds=retry_delay(item['failures']))
                if mono<item['next_start'] or started_one: continue
                blocked=admission(snapshot,spec['minimum_mb'])
                if blocked: item['state']='deferred_'+blocked; continue
                if port_open(spec['port']): item['state']='unhealthy_external_process'; continue
                label=stamp.strftime('%Y%m%dT%H%M%SZ')+'-'+spec['name']
                try:
                    with (ROOT/(label+'.stdout.log')).open('wb') as out,(ROOT/(label+'.stderr.log')).open('wb') as err:
                        child=subprocess.Popen(spec['argv'],cwd=REPO,stdout=out,stderr=err,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                except OSError as error:
                    item['failures']+=1
                    item.update(state='launch_error',next_start=mono+retry_delay(item['failures']))
                    event('service_launch_error',service=spec['name'],error=str(error),retry_seconds=retry_delay(item['failures']))
                    continue
                item.update(process=child,started=mono,state='starting'); started_one=True
                event('service_started',service=spec['name'],pid=child.pid)
            write(ROOT/'status.json',dict(updated_at=stamp.isoformat(),pid=os.getpid(),resources=snapshot,
                heavy_model_parallelism=1,heavy_model_min_available_mb=4096,
                services=[dict(name=i['spec']['name'],port=i['spec']['port'],state=i['state'],
                    pid=i['process'].pid if i['process'] else None,failures=i['failures']) for i in items]))
            time.sleep(10)


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    guard=exclusive(ROOT/'supervisor.lock')
    try: guard.__enter__()
    except PermissionError: sys.exit(0)
    try: run(locked=True)
    finally: guard.__exit__(None,None,None)
