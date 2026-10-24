# Scoped snapshot recovery fixes
```diff
--- before/scripts/train_barrier_vnext.py
+++ after/scripts/train_barrier_vnext.py
@@ -137,6 +137,7 @@
 
     if not (root/'last.pt').exists():commit()
     while state['stage']!='complete':
+        if getattr(args,'output',None):progress(Path(args.output),state['stage'],member=str(root),epoch=state['epoch']+1)
         pretraining=state['stage']=='pretrain';limit=pretrain_epochs if pretraining else args.epochs
         if state['epoch']>=limit or (not pretraining and state['stale']>=args.patience):
             state.update(stage='fit' if pretraining else 'complete',epoch=0,cursor=0,order=None,total=0.,seen=0)
@@ -282,7 +283,7 @@
 
 
 def progress(root,stage,**kwargs):
-    value=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
+    value=dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
     atomic_json(root/'progress.json',value);print('VNEXT '+json.dumps(value),flush=True)
 
 
@@ -318,11 +319,17 @@
         precision=getattr(args,'precision','fp32'),optimizer=dict(lr=.0006,weight_decay=.03,clip=1.),temperature_grid=list(TEMPERATURES))
     if (root/'protocol.json').exists():
         if json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('Frozen protocol changed')
-    else:
-        atomic_json(root/'protocol.json',protocol);snapshot=root/'source-snapshot';snapshot.mkdir(exist_ok=True)
-        for source in SOURCES:shutil.copy2(BASE/source,snapshot/Path(source).name)
+    else:atomic_json(root/'protocol.json',protocol)
+    snapshot=root/'source-snapshot';snapshot.mkdir(exist_ok=True)
     for source,digest in protocol['sources'].items():
-        if sha256(root/'source-snapshot'/Path(source).name)!=digest:raise ValueError('Source snapshot changed')
+        target=snapshot/Path(source).name
+        if not target.exists():
+            if sha256(BASE/source)!=digest:raise ValueError('Current source differs from frozen snapshot identity')
+            temporary=target.with_suffix(target.suffix+'.tmp');shutil.copy2(BASE/source,temporary)
+            with temporary.open('r+b') as stream:os.fsync(stream.fileno())
+            if sha256(temporary)!=digest:raise ValueError('Source changed while taking snapshot')
+            os.replace(temporary,target)
+        if sha256(target)!=digest:raise ValueError('Source snapshot changed')
     return protocol,cohorts
 
 
@@ -398,6 +405,7 @@
     args.context=10080;args.width=96;args.layers=3;args.batch=64;args.microbatch=16
     args.epochs=12;args.patience=3;args.pretrain_epochs=2;args.checkpoint_batches=200
     root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True);source=Path(args.source).resolve()
+    sources_at_start={p:sha256(BASE/p) for p in SOURCES}
     with run_lock(root/'run.lock'):
         torch.set_num_threads(args.threads)
         wait_source(root,source)
@@ -411,6 +419,7 @@
                 if args.device.startswith('cuda'):
                     torch.use_deterministic_algorithms(True);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
                 args.precision='bf16' if args.device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'fp32'
+                if sources_at_start!={p:sha256(BASE/p) for p in SOURCES}:raise ValueError('Source changed while queued; restart to load one consistent version')
                 protocol,cohorts=prepare(root,data,args)
                 if shutil.disk_usage(root).free<5*2**30:raise OSError('Less than5GiB free disk reserve')
                 try:run_experiment(root,data,args,protocol,cohorts)
--- before/checks/check_barrier_vnext_runner.py
+++ after/checks/check_barrier_vnext_runner.py
@@ -1,6 +1,7 @@
 """Real tiny CPU optimization and deterministic interrupted resume for vNext."""
 import importlib.util
 import json
+import os
 from pathlib import Path
 import tempfile
 from types import SimpleNamespace
@@ -106,6 +107,7 @@
         except ValueError as error:assert 'resume' in str(error).lower()
         else:raise AssertionError('Unavailable source partitions accepted')
         assert json.loads((wait_root/'progress.json').read_text())['stage']=='source_blocked'
+        assert json.loads((wait_root/'progress.json').read_text())['pid']==os.getpid()
         # Exercise the real stage coordinator with separate synthetic role cohorts.
         tiny=TinyData();tiny.metadata['folds']=[dict(name='tiny')];end_to_end=root/'end-to-end';end_to_end.mkdir()
         (end_to_end/'protocol.json').write_text('{}');original_batch=tiny.batch
@@ -125,6 +127,42 @@
         completed=json.loads((end_to_end/'results.json').read_text())
         assert set(completed['groups'])=={'tiny/baseline','tiny/pretrained'}
         assert json.loads((end_to_end/'progress.json').read_text())['stage']=='complete'
+        # Committing a protocol must not make an interrupted snapshot unrecoverable.
+        prepared=TinyData();prepared.metadata['folds']=[dict(name='snapshot',boundaries=['2026-01-01','2026-01-03','2026-01-05','2026-01-07','2026-01-09'])]
+        prepared.arrays['time_ns']=np.array([np.datetime64('2026-01-01','ns').astype(np.int64)+int(days*86_400_000_000_000)
+            for days in (0,.1,2,2.1,4,4.1,6,6.1,8,8.1)],np.int64)
+        prepared.eligible_history=lambda _:np.ones(10,bool)
+        snapshot_root=root/'snapshot-resume';snapshot_root.mkdir();write_json=runner.atomic_json
+        def interrupt_protocol(path,value):
+            write_json(path,value)
+            if Path(path).name=='protocol.json':raise InterruptedError('Committed protocol before snapshots')
+        with patch.object(runner,'atomic_json',interrupt_protocol):
+            try:runner.prepare(snapshot_root,prepared,args)
+            except InterruptedError:pass
+            else:raise AssertionError('Protocol interruption not triggered')
+        protocol,_=runner.prepare(snapshot_root,prepared,args)
+        for name,digest in protocol['sources'].items():
+            assert runner.sha256(snapshot_root/'source-snapshot'/Path(name).name)==digest
+        # A torn temporary copy may be replaced; completed corrupt snapshots may not.
+        target=snapshot_root/'source-snapshot'/Path(runner.SOURCES[1]).name;target.unlink()
+        def interrupt_copy(source,target_path):
+            Path(target_path).write_bytes(b'torn copy');raise InterruptedError('Interrupted snapshot copy')
+        with patch.object(runner.shutil,'copy2',interrupt_copy):
+            try:runner.prepare(snapshot_root,prepared,args)
+            except InterruptedError:pass
+            else:raise AssertionError('Snapshot-copy interruption not triggered')
+        runner.prepare(snapshot_root,prepared,args)
+        target.write_bytes(b'corrupt completed snapshot')
+        try:runner.prepare(snapshot_root,prepared,args)
+        except ValueError:pass
+        else:raise AssertionError('Corrupt completed snapshot silently repaired')
+        target.unlink()
+        frozen_hash=runner.sha256
+        def changed_source(path):return 'changed' if Path(path)==runner.BASE/runner.SOURCES[0] else frozen_hash(path)
+        with patch.object(runner,'sha256',changed_source):
+            try:runner.prepare(snapshot_root,prepared,args)
+            except ValueError:pass
+            else:raise AssertionError('Changed current source accepted while recovering snapshots')
         assert not list(root.rglob('*predictions*.npz')), 'Do not duplicate huge per-query test arrays'
     print('PASS vNext actual pretraining/fit/calibration,128queries,exact fit+pretrain resume,source/cohort identity,3seed mixture,streaming metrics and preCUDA wait')
 

```
