"""DiT worker (NPU env): 指定層のセッションを1回だけ作り、全組の cur を通して保存して終わる。
出力は一時名で書いてから正式名に変え、完了の印 (hash と CPU 時間) を書く。"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import stage_dit as sd
import prepost as rn
import layer_var as lv
import layer_hd as lh

def cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', required=True)
    ap.add_argument('--intra', type=int, default=2)
    ap.add_argument('--no-spin', action='store_true')
    ap.add_argument('--no-arena', action='store_true')
    ap.add_argument('--size', type=int, choices=[512, 1024], default=512)
    ap.add_argument('--run-id', required=True)
    a = ap.parse_args()
    # わざと失敗させる検証用: 環境変数があれば非0で終わる (出力は作らない)
    if os.environ.get('MOSSNP_FAIL_WORKER'):
        print('worker forced failure (MOSSNP_FAIL_WORKER)', flush=True)
        raise SystemExit(7)
    if a.run_id not in Path(a.inp).name or a.run_id not in Path(a.out).name:
        raise SystemExit('run_id が入出力名に含まれていない')
    t_start = time.perf_counter(); c_start = cpu_seconds()
    layer_mod = lv if a.size == 512 else lh
    run_layer = lv.run_layer_var if a.size == 512 else lh.run_layer_hd
    lo, hi = a.layers.split('-'); layers = list(range(int(lo), int(hi) + 1))
    z = np.load(a.inp)
    n = int(z['n_items'])
    consts_by_i = {}
    for i in range(n):
        L_i = int(z[f'L{i}'])
        consts_by_i[i] = layer_mod.consts_for(z[f'sin{i}'].astype(np.float32), z[f'cos{i}'].astype(np.float32), L_i)
    curs = {i: z[f'cur{i}'].astype(np.float32) for i in range(n)}
    mods = {i: {nm: z[f'mod{i}_{nm}'].astype(np.float32) for nm in rn.BLOCK_MODS} for i in range(n)}
    z.close()
    opts = argparse.Namespace(intra=a.intra, no_spin=a.no_spin, no_arena=a.no_arena, size=a.size)
    sessions, each, ss, fail = sd.create_sessions_p5(layers, opts)
    if sessions is None:
        raise SystemExit(f'worker create failed at {fail}')
    print(f'worker size {a.size} layers {layers[0]}-{layers[-1]} created {ss:.2f}s', flush=True)
    out_data = {}
    items_s = []
    for i in range(n):
        t_i = time.perf_counter()
        cur = curs[i]
        for l in layers:
            cur, _, _ = run_layer(sessions, l, cur, mods[i], consts_by_i[i])
        items_s.append(time.perf_counter() - t_i)
        if cur.dtype != np.float32 or not bool(np.isfinite(cur).all()):
            raise SystemExit(f'worker 出力が有限でない item {i}')
        out_data[f'cur{i}'] = cur
    tmp_out = Path(str(a.out) + '.tmp.npz')
    np.savez(tmp_out, **out_data)
    os.replace(tmp_out, a.out)
    cpu = cpu_seconds() - c_start; wall = time.perf_counter() - t_start
    sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    done = {'run_id': a.run_id, 'out': str(a.out), 'sha256': sha, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2),
            'layers': a.layers, 'size': a.size, 'create_s': round(ss, 2), 'items_s': [round(t, 3) for t in items_s]}
    tmp_done = Path(str(a.out) + '.done.json.tmp')
    tmp_done.write_text(json.dumps(done), encoding='utf-8')
    os.replace(tmp_done, str(a.out) + '.done.json')
    print(f'worker done saved {a.out} cpu {cpu:.2f}s', flush=True)

if __name__ == '__main__':
    main()
