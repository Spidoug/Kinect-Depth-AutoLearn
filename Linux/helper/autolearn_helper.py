#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
from pathlib import Path

DATA_HOME = Path(os.environ.get('XDG_DATA_HOME', str(Path.home()/'.local/share'))) / 'kinect-depth-autolearn'
STATUS = DATA_HOME / 'helper_status.json'
REQUEST = DATA_HOME / 'training_request.json'
RUNTIME_PY = DATA_HOME / 'runtime_python.txt'
BIN_DIR = DATA_HOME / 'bin'
LOG = DATA_HOME / 'helper.log'
RUNNING = True


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def log(msg):
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def stop(*_):
    global RUNNING
    RUNNING = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

bridges = {}
train_proc = None
train_req_id = ''


def ensure_bridge(name, port):
    exe = BIN_DIR / name
    p = bridges.get(name)
    if p is not None and p.poll() is None:
        return p
    if not exe.is_file():
        bridges[name] = None
        return None
    try:
        out = (DATA_HOME / f'{name}.log').open('ab', buffering=0)
        p = subprocess.Popen([str(exe), str(port)], stdout=out, stderr=subprocess.STDOUT, cwd=str(DATA_HOME), start_new_session=True)
        bridges[name] = p
        log(f'started {name} pid={p.pid} port={port}')
        return p
    except Exception as e:
        bridges[name] = None
        log(f'failed to start {name}: {e!r}')
        return None


def set_progress_error(dataset, message):
    try:
        out = Path(dataset) / 'output'
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out/'progress.json', {
            'state':'error','progress':0.0,'epoch':0,'epochs':90,'batch':0,'batches':0,
            'loss':0.0,'val':0.0,'best':0.0,'eta':'--:--:--','elapsed':'00:00:00',
            'device':'','gpu':'','error':message
        })
    except Exception as e:
        log(f'could not write progress error: {e!r}')


def launch_training(req):
    global train_proc, train_req_id
    dataset = req['dataset']
    trainer = req['trainer']
    py = req.get('python','').strip()
    if not py and RUNTIME_PY.is_file():
        py = RUNTIME_PY.read_text(encoding='utf-8').splitlines()[0].strip()
    if not py:
        raise RuntimeError('runtime Python is not configured')
    cmd = [py, trainer, dataset,
           '--progress-ms', str(req.get('progress_ms',100)),
           '--log', req['session_log'],
           '--resume', 'auto',
           '--checkpoint-batches', str(req.get('checkpoint_batches',10)),
           '--console-seconds', str(req.get('console_seconds',15))]
    mirror = Path(req['mirror_log'])
    mirror.parent.mkdir(parents=True, exist_ok=True)
    out = mirror.open('ab', buffering=0)
    env = os.environ.copy(); env['PYTHONUNBUFFERED']='1'
    train_proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    train_req_id = req.get('request_id','')
    log(f'training started pid={train_proc.pid} request={train_req_id} dataset={dataset}')


def consume_request():
    global train_proc
    if train_proc is not None and train_proc.poll() is None:
        return
    if not REQUEST.is_file():
        return
    try:
        req = json.loads(REQUEST.read_text(encoding='utf-8'))
        REQUEST.unlink(missing_ok=True)
        launch_training(req)
    except Exception as e:
        log(f'training launch failed: {e!r}')
        try:
            if 'req' in locals():
                set_progress_error(req.get('dataset',''), f'Linux helper could not start training: {e}')
        except Exception:
            pass


def write_status():
    global train_proc
    tp = train_proc
    atomic_json(STATUS, {
        'version':1,
        'heartbeat_ms':int(time.time()*1000),
        'pid':os.getpid(),
        'remold_bridge': bool(bridges.get('remold_bridge') and bridges['remold_bridge'].poll() is None),
        'openkinect_bridge': bool(bridges.get('openkinect_bridge') and bridges['openkinect_bridge'].poll() is None),
        'training': bool(tp and tp.poll() is None),
        'training_pid': tp.pid if tp and tp.poll() is None else 0,
        'training_request_id': train_req_id,
    })


def main():
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    log('helper started')
    while RUNNING:
        ensure_bridge('remold_bridge',38174)
        ensure_bridge('openkinect_bridge',38173)
        consume_request()
        write_status()
        time.sleep(0.5)
    for p in list(bridges.values()):
        if p and p.poll() is None:
            try: p.terminate()
            except Exception: pass
    if train_proc and train_proc.poll() is None:
        log('helper stopping while training remains active; leaving trainer running')
    log('helper stopped')

if __name__ == '__main__':
    main()
