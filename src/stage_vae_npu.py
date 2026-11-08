"""VAE段 (NPU, 1024^2): AMD klein の DD 済みデコーダで潜在 -> PNG。
書き出しが全部済んでから完了の印を書き、親がそれを確かめる。最後は TerminateProcess で終える。"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import npu_config as cfg

LATENT_SHAPE = (1, 128, 64, 64)

def cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system

def bn_stats():
    from safetensors import safe_open
    moss_vae = cfg.moss() / 'vae'
    eps = float(json.loads((moss_vae / 'config.json').read_text(encoding='utf-8'))['batch_norm_eps'])
    with safe_open(str(moss_vae / 'diffusion_pytorch_model.safetensors'), framework='np') as f:
        mean = f.get_tensor('bn.running_mean').astype(np.float32)
        var = f.get_tensor('bn.running_var').astype(np.float32)
    return mean, np.sqrt(var + np.float32(eps)).astype(np.float32)

def to_vae_input(lat, mean, std):
    z = lat * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return np.ascontiguousarray(z, np.float32)

def to_uint8(image):
    return (np.clip(image[0].transpose(1, 2, 0) / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)

def make_session(intra):
    import onnxruntime
    genai = cfg.genai_sd()
    sys.path.insert(0, str(genai))
    from src.utils import common
    os.environ.setdefault('DD_ROOT', str(genai / 'lib'))
    os.environ.setdefault('DD_PLUGINS_ROOT', str(genai / 'lib' / 'transaction' / 'stx'))
    klein = cfg.klein_vae_dd()
    custom_op = cfg.ryzen_dll()
    so = common.config_session_options(str(custom_op), klein, False, -1.0, [common.EP_NAME, 'CPUExecutionProvider'])
    if intra:
        so.intra_op_num_threads = intra
        so.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(klein / 'replaced.onnx'), sess_options=so)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--intra', type=int, default=1)
    ap.add_argument('--size', type=int, choices=[1024], default=1024)
    ap.add_argument('--run-id', required=True)
    a = ap.parse_args()
    t_start = time.perf_counter(); c_start = cpu_seconds()
    out = Path(a.out)
    entries = json.loads((out / 'text_manifest.json').read_text(encoding='utf-8'))
    if entries.get('run_id') != a.run_id:
        raise SystemExit('text_manifest の run_id が違う')
    entries = entries['items']
    done_items = []
    log = {'run_id': a.run_id, 'size': a.size, 'decoder': 'klein-dd-npu', 'intra': a.intra, 'items': []}
    if entries:
        from PIL import Image
        mean, std = bn_stats()
        t0 = time.perf_counter()
        sess = make_session(a.intra)
        load_s = time.perf_counter() - t0
        log['load_s'] = round(load_s, 2)
        in_name = sess.get_inputs()[0].name
        print(f'vae(npu) load {load_s:.2f}s intra {a.intra} items {len(entries)}', flush=True)
        for e in entries:
            name = f"{e['id']}_s{e['seed']}"
            lat = np.load(out / 'latents' / f'{name}.npz')['final_latents']
            if lat.shape != LATENT_SHAPE or lat.dtype != np.float32 or not bool(np.isfinite(lat).all()):
                raise SystemExit(f'潜在の検査に失敗: {name} shape={lat.shape} dtype={lat.dtype}')
            lat = lat.astype(np.float32)
            t1 = time.perf_counter(); c1 = cpu_seconds()
            img = sess.run(None, {in_name: to_vae_input(lat, mean, std)})[0]
            run_s = time.perf_counter() - t1
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32)
            finite = bool(np.isfinite(img).all())
            if not finite:
                raise SystemExit(f'デコーダ出力が有限でない: {name}')
            img_path = out / f'{name}.png'
            tmp_path = out / f'{name}.tmp.png'
            Image.fromarray(to_uint8(img.astype(np.float32))).save(tmp_path)
            os.replace(tmp_path, img_path)
            dt, dc = time.perf_counter() - t1, cpu_seconds() - c1
            sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
            log['items'].append({'id': e['id'], 'seed': e['seed'], 'png': str(img_path), 'finite': True, 'sha256': sha,
                                 'run_s': round(run_s, 3), 'decode_s': round(dt, 2), 'decode_cpu_s': round(dc, 2)})
            done_items.append({'id': e['id'], 'seed': e['seed'], 'png': img_path.name, 'sha256': sha})
            print(f'vae(npu) saved {img_path} run {run_s:.2f}s total {dt:.2f}s finite True', flush=True)
    wall = time.perf_counter() - t_start; cpu = cpu_seconds() - c_start
    log['wall_s'] = round(wall, 2); log['cpu_s'] = round(cpu, 2)
    tmp_log = out / 'vae_log.json.tmp'
    tmp_log.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp_log, out / 'vae_log.json')
    (out / f'vae_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'vae', 'run_id': a.run_id, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2)}), encoding='utf-8')
    marker = {'run_id': a.run_id, 'items': done_items, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2)}
    tmp_marker = out / f'vae_done_{a.run_id}.json.tmp'
    tmp_marker.write_text(json.dumps(marker, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp_marker, out / f'vae_done_{a.run_id}.json')
    print(f'vae(npu) done wall {wall:.2f}s cpu {cpu:.2f}s', flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k32.TerminateProcess(k32.GetCurrentProcess(), 0)

if __name__ == '__main__':
    main()
