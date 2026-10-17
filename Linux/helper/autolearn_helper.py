
import json, os, signal, subprocess, sys, tempfile, time
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
    # A unique temporary file avoids collisions with concurrent writers.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}.', suffix='.tmp',
                                     delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
bridge_retry_after = {}
BRIDGE_RETRY_SECONDS = 5.0
train_proc = None
train_req_id = ''


def ensure_bridge(name, port):
    exe = BIN_DIR / name
    p = bridges.get(name)
    if time.monotonic() < bridge_retry_after.get(name, 0):
        return p if p is not None and p.poll() is None else None
    if p is not None and p.poll() is None:
        return p
    if not exe.is_file():
        bridge_retry_after[name] = time.monotonic() + BRIDGE_RETRY_SECONDS
        bridges[name] = None
        return None
    try:
        out = (DATA_HOME / f'{name}.log').open('ab', buffering=0)
        try:
            p = subprocess.Popen([str(exe), str(port)], stdout=out, stderr=subprocess.STDOUT, cwd=str(DATA_HOME), start_new_session=True)
        finally:
            out.close()  # The child keeps its duplicated descriptor.
        bridges[name] = p
        bridge_retry_after[name] = time.monotonic() + BRIDGE_RETRY_SECONDS
        log(f'started {name} pid={p.pid} port={port}')
        return p
    except Exception as e:
        bridge_retry_after[name] = time.monotonic() + BRIDGE_RETRY_SECONDS
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
    if not isinstance(req, dict):
        raise ValueError('training request must be a JSON object')
    dataset = req['dataset']
    trainer = req['trainer']
    if not isinstance(dataset, str) or not dataset.strip() or not Path(dataset).is_dir():
        raise ValueError('training dataset directory does not exist')
    if not isinstance(trainer, str) or not Path(trainer).is_file():
        raise ValueError('training script does not exist')
    py = req.get('python','').strip()
    if not py and RUNTIME_PY.is_file():
        py = next(iter(RUNTIME_PY.read_text(encoding='utf-8').splitlines()), '').strip()
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
    try:
        train_proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    finally:
        out.close()  # Prevent file-descriptor leaks across restarts.
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
