import argparse, json, time
from pathlib import Path

def latest(base):
    root=base/'dataset'; items=[p for p in root.glob('session_*') if (p/'labels.csv').is_file()]
    if not items: raise SystemExit('No dataset session found.')
    return max(items,key=lambda p:p.name)

def fmt(v): return '--' if v is None else str(v)

ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['pause','resume','status']); ap.add_argument('dataset',nargs='?',default=''); a=ap.parse_args()
root=Path(a.dataset).resolve() if a.dataset else latest(Path.cwd()); out=root/'output'; out.mkdir(parents=True,exist_ok=True)
if a.action in ('pause','resume'):
    paused=a.action=='pause'; p=out/'control.json'; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps({'paused':paused,'updated':time.time()}),encoding='utf-8'); tmp.replace(p)
    print(('Pause requested' if paused else 'Resume requested')+f': {root}')
else:
    pp=out/'progress.json'
    if not pp.is_file(): print(f'No progress.json yet: {root}'); raise SystemExit(0)
    j=json.loads(pp.read_text(encoding='utf-8'))
    print(f'Dataset : {root}')
    print(f"State   : {j.get('state','?')}")
    print(f"Progress: {float(j.get('progress',0))*100:.2f}%")
    print(f"Epoch   : {j.get('epoch',0)}/{j.get('epochs',0)}")
    print(f"Batch   : {j.get('batch',0)}/{j.get('batches',0)}")
    print(f"Loss    : {j.get('loss',0)}   Val: {j.get('val',0)}   Best: {j.get('best',0)}")
    print(f"Elapsed : {j.get('elapsed','--:--:--')}   ETA: {j.get('eta','--:--:--')}")
    print(f"Device  : {j.get('device','')} {j.get('gpu','')}")
