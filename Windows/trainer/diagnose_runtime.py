from pathlib import Path
import json, sys, os

def fail(code, msg):
    print('[ERROR]', msg)
    raise SystemExit(code)

def main():
    root = Path(__file__).resolve().parent.parent
    status_path = root / 'runtime_status.json'
    if not status_path.is_file() and os.environ.get('LOCALAPPDATA'):
        status_path = Path(os.environ['LOCALAPPDATA'])/'KinectDepthAutoLearn'/'runtime_status.json'
    if not status_path.is_file():
        fail(2, 'runtime_status.json is missing. Run install_autolearn.bat.')
    try:
        status = json.loads(status_path.read_text(encoding='utf-8-sig'))
    except Exception as e:
        fail(2, f'invalid runtime_status.json: {e}')
    backend = str(status.get('backend','')).upper()
    if backend not in {'CPU','CUDA','DIRECTML'}:
        fail(2, f'invalid backend in runtime status: {backend!r}')

    print('Python:', sys.version.split()[0])
    print('Selected backend:', backend)
    try:
        import torch
    except Exception as e:
        fail(3, f'PyTorch: {type(e).__name__}: {e}')
    print('PyTorch:', torch.__version__)

    try:
        if backend == 'CPU':
            d = torch.device('cpu')
            y = (torch.tensor([2.0, 3.0], device=d) * 2 + 1).tolist()
            if y != [5.0, 7.0]: fail(4, f'teste CPU inesperado: {y}')
            print('CPU: OK')
        elif backend == 'CUDA':
            if not torch.cuda.is_available(): fail(4, 'CUDA selected, but torch.cuda.is_available()=False')
            d = torch.device('cuda')
            y = (torch.tensor([2.0, 3.0], device=d) * 2 + 1).cpu().tolist()
            if y != [5.0, 7.0]: fail(4, f'teste CUDA inesperado: {y}')
            print('CUDA: OK', torch.cuda.get_device_name(0))
        else:
            import torch_directml
            d = torch_directml.device()
            y = (torch.tensor([2.0, 3.0], device=d) * 2 + 1).cpu().tolist()
            if y != [5.0, 7.0]: fail(4, f'teste DirectML inesperado: {y}')
            print('DirectML: OK', d)
    except SystemExit:
        raise
    except Exception as e:
        fail(4, f'{backend} is not functional: {type(e).__name__}: {e}')

    for name in ('numpy','PIL','onnx','onnxruntime'):
        try:
            m=__import__(name)
            print(name, getattr(m,'__version__','OK'))
        except Exception as e:
            fail(5, f'{name}: {type(e).__name__}: {e}')

    sketch=root/'app'/'KinectDepthAutoLearn'
    required=[sketch/'data'/'movenet.onnx',sketch/'data'/'rtmpose-hand.onnx']
    jars=list((sketch/'code').glob('onnxruntime-*.jar')) if (sketch/'code').exists() else []
    for p in required:
        if not p.is_file() or p.stat().st_size < 100_000:
            fail(6, f'missing/invalid file: {p}')
        print('File:', p.name, p.stat().st_size, 'bytes')
    if not jars:
        fail(6,'ONNX Runtime Java JAR is missing from app/KinectDepthAutoLearn/code/.')
    print('JAR:', jars[0].name)
    print(f'[OK] Runtime {backend} functional.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
