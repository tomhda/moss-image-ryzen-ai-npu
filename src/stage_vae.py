"""VAE段 (text env, CPU): 最後の潜在 npz 群 -> PNG。潜在とデコーダ出力を保存前に確かめる。"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import npu_config as cfg

def cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--size', type=int, choices=[512, 1024], default=512)
    ap.add_argument('--run-id', required=True)
    a = ap.parse_args()
    t_start = time.perf_counter(); c_start = cpu_seconds()
    try:
        torch.set_num_threads(a.threads)
        torch.set_grad_enabled(False)
        out = Path(a.out)
        manifest = json.loads((out / 'text_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('run_id') != a.run_id:
            raise SystemExit('text_manifest の run_id が違う')
        entries = manifest['items']
        log = {'run_id': a.run_id, 'size': a.size, 'threads': a.threads, 'items': []}
        if entries:
            from diffusers import AutoencoderKLFlux2
            from diffusers.image_processor import VaeImageProcessor
            t0 = time.perf_counter()
            vae = AutoencoderKLFlux2.from_pretrained(cfg.moss() / 'vae', torch_dtype=torch.float32).eval()
            load_s = time.perf_counter() - t0
            log['load_s'] = round(load_s, 2)
            print(f'vae load {load_s:.2f}s threads {torch.get_num_threads()} items {len(entries)} size {a.size}', flush=True)
            hw = 32 if a.size == 512 else 64
            for e in entries:
                name = f"{e['id']}_s{e['seed']}"
                z = np.load(out / 'latents' / f'{name}.npz')
                lat_np = z['final_latents']
                if lat_np.shape != (1, 128, hw, hw) or lat_np.dtype != np.float32 or not bool(np.isfinite(lat_np).all()):
                    raise SystemExit(f'潜在の検査に失敗: {name} shape={lat_np.shape} dtype={lat_np.dtype}')
                lat = torch.from_numpy(lat_np.astype(np.float32))
                t1 = time.perf_counter(); c1 = cpu_seconds()
                with torch.no_grad():
                    zz = lat.to(dtype=vae.dtype)
                    bn = vae.bn; eps = vae.config.batch_norm_eps
                    z_dn = zz * (bn.running_var + eps).sqrt().to(zz).view(1, -1, 1, 1) + bn.running_mean.to(zz).view(1, -1, 1, 1)
                    b, c, h, w = z_dn.shape
                    vae_in = z_dn.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
                    decoded = vae.decode(vae_in, return_dict=False)[0].clamp(-1, 1)
                if not bool(torch.isfinite(decoded).all()):
                    raise SystemExit(f'デコーダ出力が有限でない: {name}')
                img = VaeImageProcessor(vae_scale_factor=16).postprocess(decoded, output_type='pil')[0]
                img_path = out / f'{name}.png'
                tmp_path = out / f'{name}.tmp.png'
                img.save(tmp_path)
                os.replace(tmp_path, img_path)
                dt, dc = time.perf_counter() - t1, cpu_seconds() - c1
                log['items'].append({'id': e['id'], 'seed': e['seed'], 'png': str(img_path),
                                     'decode_s': round(dt, 2), 'decode_cpu_s': round(dc, 2)})
                print(f'vae saved {img_path} {dt:.2f}s', flush=True)
        wall = time.perf_counter() - t_start; cpu = cpu_seconds() - c_start
        log['wall_s'] = round(wall, 2); log['cpu_s'] = round(cpu, 2)
        tmp_log = out / 'vae_log.json.tmp'
        tmp_log.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp_log, out / 'vae_log.json')
        (out / f'vae_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'vae', 'run_id': a.run_id, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2)}), encoding='utf-8')
        print(f'vae done wall {wall:.2f}s cpu {cpu:.2f}s', flush=True)
    except BaseException:
        try:
            (Path(a.out) / f'vae_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'vae', 'run_id': a.run_id, 'cpu_s': round(cpu_seconds() - c_start, 2), 'failed': True}), encoding='utf-8')
        except Exception:
            pass
        raise

if __name__ == '__main__':
    main()
