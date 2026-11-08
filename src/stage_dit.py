"""DiT段 (NPU env): 全組4ステップ、最後の潜在を npz 保存。--size 512|1024。
512 は A が replaced_var、1024 は A が attn1024。T は共通の FFN の塊。
worker は実行ごとの一時フォルダを使い、run_id で今回の分だけ受け取る。"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import prepost as rn
import layer_var as lv
import layer_hd as lh
from moss_model import load_pipeline_module
import npu_config as cfg

def cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system

def create_sessions_p5(layers, opts, watch_gb=2.5):
    size = getattr(opts, 'size', 512)
    sessions, each = {}, {}
    t_all = time.perf_counter()
    for l in layers:
        a_name = f'attn_L{l}' if size == 512 else f'attn1024_L{l}'
        for tag, pname in ((f'{l}:A', a_name), (f'{l}:T', f'tile_nc_L{l}')):
            t = time.perf_counter()
            if size == 1024 and tag.endswith(':A'):
                sessions[tag] = lh.dd_session_a1024(pname, opts.intra, opts.no_spin, opts.no_arena)
            else:
                sessions[tag] = lv.dd_session_var(pname, opts.intra, opts.no_spin, opts.no_arena)
            each[tag] = time.perf_counter() - t
            avail = rn.mem_snapshot().get('avail_mb')
            print(f'session {tag} {pname}: {each[tag]:.2f}s avail {avail:.0f}MB' if avail else f'session {tag} {each[tag]:.2f}s', flush=True)
            if avail is not None and avail < watch_gb * 1024:
                print(f'AVAIL LOW {avail:.0f}MB < {watch_gb}GB at {tag}', flush=True)
                return None, each, time.perf_counter() - t_all, l
    return sessions, each, time.perf_counter() - t_all, None

def rope_tables(gen, text_len, size=512):
    hw = 32 if size == 512 else 64
    rope = gen['rope_mod'](hw, hw, text_len, torch.device('cpu')).numpy()
    return rope[0, :, :64], rope[0, :, 64:]

def consts_for_L(gen, text_len, size=512):
    sin, cos = rope_tables(gen, text_len, size)
    return lv.consts_for(sin, cos, text_len) if size == 512 else lh.consts_for(sin, cos, text_len)

def make_scheduler():
    sch = load_pipeline_module(cfg.moss()).MOSSImageScheduler()
    sch.set_timesteps(4, device='cpu')
    return sch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--intra', type=int, default=2)
    ap.add_argument('--no-spin', action='store_true')
    ap.add_argument('--no-arena', action='store_true')
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--group', choices=['auto', 'all', 'g2', 'g3'], default='auto')
    ap.add_argument('--size', type=int, choices=[512, 1024], default=512)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--_reexec', action='store_true')
    a = ap.parse_args()
    t_start = time.perf_counter(); c_start = cpu_seconds()
    worker_cpu_total = 0.0
    try:
        HW = 32 if a.size == 512 else 64
        IMGTOK = HW * HW
        LAYER = lv if a.size == 512 else lh
        RUN_LAYER = lv.run_layer_var if a.size == 512 else lh.run_layer_hd
        if a.torch_threads:
            torch.set_num_threads(a.torch_threads)
        torch.set_grad_enabled(False)
        out = Path(a.out)
        manifest = json.loads((out / 'text_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('run_id') != a.run_id:
            raise SystemExit(f"text_manifest の run_id が違う: {manifest.get('run_id')} != {a.run_id}")
        entries = manifest['items']
        print(f'dit size {a.size} items {len(entries)} failed {len(manifest.get("failed", []))} intra {a.intra} no_spin {a.no_spin} no_arena {a.no_arena} torch_threads {torch.get_num_threads()}', flush=True)
        print(f'mem before {rn.mem_snapshot()}', flush=True)
        log = {'run_id': a.run_id, 'size': a.size, 'config': {'intra': a.intra, 'no_spin': a.no_spin, 'no_arena': a.no_arena,
                          'torch_threads': torch.get_num_threads(), 'group': a.group},
               'items': [], 'worker_runs': []}
        if entries:
            gen = rn.load_prepost()
            _rope_cache = {}
            def tables_for(L):
                if L not in _rope_cache:
                    _rope_cache[L] = rope_tables(gen, L, a.size)
                return _rope_cache[L]
            def consts_for_key(e):
                sin, cos = tables_for(int(e['tokens']))
                return (lv if a.size == 512 else lh).consts_for(sin, cos, int(e['tokens']))
            sch0 = make_scheduler()
            print(f'scheduler timesteps {sch0.timesteps.tolist()} sigmas {sch0.sigmas.tolist()}', flush=True)
            timesteps = [float(t) for t in sch0.timesteps.tolist()]
            layers = list(range(42))
            groups, grouped, sessions, sess_s = [layers], False, None, 0.0
            avail0 = rn.mem_snapshot().get('avail_mb') or 0
            if a.size == 1024:
                print(f'1024: avail {avail0:.0f}MB, grouped g3 (14 x 3)', flush=True)
                groups, grouped = [layers[:14], layers[14:28], layers[28:]], True
            elif a.group == 'g2':
                groups, grouped = [layers[:21], layers[21:]], True
            elif a.group == 'g3':
                groups, grouped = [layers[:14], layers[14:28], layers[28:]], True
            elif a.group == 'all':
                pass
            elif a.group == 'auto' and avail0 < 15000:
                print(f'auto: avail {avail0:.0f}MB < 15000MB, grouped g2 from start', flush=True)
                groups, grouped = [layers[:21], layers[21:]], True
            # メモリ見込み: 一番大きい組の A+T 組数 x (作成+初回実行) を始める前に確かめる
            per_pair_mb = 750 if a.size == 1024 else 350
            largest = max(len(g) for g in groups) if grouped else 42
            need_mb = largest * per_pair_mb + 1024
            avail_now = rn.mem_snapshot().get('avail_mb') or 0
            print(f'memory check: largest_group_pairs {largest} per_pair {per_pair_mb}MB need {need_mb:.0f}MB avail {avail_now:.0f}MB', flush=True)
            if avail_now < need_mb:
                raise SystemExit(f'メモリ不足: 空き {avail_now:.0f}MB < 見込み {need_mb:.0f}MB (最大組 {largest} 組 x {per_pair_mb}MB + 予備 1024MB)')
            if not grouped and a.group != 'all':
                sessions, each, sess_s, fail_at = create_sessions_p5(layers, a)
                if sessions is None:
                    if a._reexec:
                        raise SystemExit(f'all sessions failed at layer {fail_at} (reexec済み)')
                    import subprocess
                    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--items', a.items, '--out', a.out,
                           '--intra', str(a.intra), '--group', 'g2', '--size', str(a.size), '--run-id', a.run_id, '--_reexec']
                    if a.no_spin: cmd.append('--no-spin')
                    if a.no_arena: cmd.append('--no-arena')
                    if a.torch_threads: cmd += ['--torch-threads', str(a.torch_threads)]
                    print(f'all sessions failed at layer {fail_at}, reexec grouped: {cmd}', flush=True)
                    rc = subprocess.run(cmd).returncode
                    if rc != 0:
                        raise SystemExit(f'grouped reexec failed exit {rc}')
                    # reexec 側が dit_log と cpu json を書いたので引き継ぐ
                    sub_cpu = out / f'dit_cpu_{a.run_id}.json'
                    if sub_cpu.exists():
                        worker_cpu_total = float(json.loads(sub_cpu.read_text(encoding='utf-8')).get('worker_cpu_s', 0.0))
                    return
            elif not grouped and a.group == 'all':
                sessions, each, sess_s, fail_at = create_sessions_p5(layers, a)
                if sessions is None:
                    raise SystemExit(f'all sessions failed at layer {fail_at}')
            if sessions is not None:
                print(f'sessions all created {sess_s:.2f}s mem {rn.mem_snapshot()}', flush=True)
            log.update({'session_s': sess_s, 'timesteps': timesteps,
                        'config': {**log['config'], 'grouped': grouped, 'groups': [len(g) for g in groups]}})
            lat_dir = out / 'latents'; lat_dir.mkdir(parents=True, exist_ok=True)
            run_dir = out / '.run' / a.run_id / 'worker'
            run_dir.mkdir(parents=True, exist_ok=True)
            states = {}
            for e in entries:
                from safetensors import safe_open
                with safe_open(e['cache_file'], framework='pt') as f:
                    embeds = f.get_tensor('last_hidden_state')[None]
                g = torch.Generator('cpu').manual_seed(int(e['seed']))
                lat = torch.randn((1, 128, HW, HW), generator=g, dtype=torch.float32)
                states[f"{e['id']}_s{e['seed']}"] = {'entry': e, 'embeds': embeds, 'lat': lat, 'sch': make_scheduler(), 'wall': 0.0}
            t_dit0 = time.perf_counter()
            for step_idx, ts in enumerate(timesteps):
                xs, modss, heads = {}, {}, {}
                for e in entries:
                    st = states[f"{e['id']}_s{e['seed']}"]
                    x0_t, mod_t, head_t, _rope = rn.preprocess_np(gen, st['lat'], ts, st['embeds'], height=HW, width=HW)
                    xs[f"{e['id']}_s{e['seed']}"] = x0_t.numpy().astype(np.float32)
                    modss[f"{e['id']}_s{e['seed']}"] = rn.mods_of(mod_t.numpy().astype(np.float32))
                    heads[f"{e['id']}_s{e['seed']}"] = head_t
                if not grouped:
                    for e in entries:
                        t0 = time.perf_counter()
                        cur = xs[f"{e['id']}_s{e['seed']}"]
                        for l in layers:
                            cur, _, _ = RUN_LAYER(sessions, l, cur, modss[f"{e['id']}_s{e['seed']}"], consts_for_key(e))
                        vel_t = rn.postprocess_np(gen, torch.from_numpy(cur), heads[f"{e['id']}_s{e['seed']}"], height=HW, width=HW)
                        st = states[f"{e['id']}_s{e['seed']}"]
                        st['lat'] = st['sch'].step(vel_t, torch.tensor(ts), st['lat'], return_dict=False)[0].float()
                        st['wall'] += time.perf_counter() - t0
                        print(f"dit {e['id']} step {step_idx} done", flush=True)
                else:
                    curs = dict(xs)
                    import subprocess
                    for grp in groups:
                        win = run_dir / f'in_{a.run_id}_s{step_idx}_L{grp[0]}-{grp[-1]}.npz'
                        wout = run_dir / f'out_{a.run_id}_s{step_idx}_L{grp[0]}-{grp[-1]}.npz'
                        done = Path(str(wout) + '.done.json')
                        for p in (wout, done):
                            p.unlink(missing_ok=True)
                        data = {'n_items': np.array(len(entries))}
                        for i, e in enumerate(entries):
                            sin_i, cos_i = tables_for(int(e['tokens']))
                            data[f'L{i}'] = np.array(int(e['tokens']))
                            data[f'sin{i}'] = np.ascontiguousarray(sin_i, np.float32)
                            data[f'cos{i}'] = np.ascontiguousarray(cos_i, np.float32)
                        for i, e in enumerate(entries):
                            data[f'cur{i}'] = curs[f"{e['id']}_s{e['seed']}"]
                            for nm, arr in modss[f"{e['id']}_s{e['seed']}"].items():
                                data[f'mod{i}_{nm}'] = arr
                        tmp_in = Path(str(win) + '.tmp.npz')
                        np.savez(tmp_in, **data)
                        os.replace(tmp_in, win)
                        cmd = [sys.executable, '-u', str(Path(__file__).resolve().parent / 'stage_dit_worker.py'),
                               '--in', str(win), '--out', str(wout), '--layers', f'{grp[0]}-{grp[-1]}',
                               '--intra', str(a.intra), '--size', str(a.size), '--run-id', a.run_id]
                        if a.no_spin: cmd.append('--no-spin')
                        if a.no_arena: cmd.append('--no-arena')
                        t_w = time.perf_counter()
                        rc = subprocess.run(cmd).returncode
                        w_wall = time.perf_counter() - t_w
                        w_cpu = 0.0
                        if done.exists():
                            try:
                                dj = json.loads(done.read_text(encoding='utf-8'))
                                if dj.get('run_id') == a.run_id:
                                    w_cpu = float(dj.get('cpu_s', 0.0))
                                    if hashlib.sha256(wout.read_bytes()).hexdigest() != dj.get('sha256'):
                                        rc = -1
                                        print('worker output hash mismatch', flush=True)
                            except Exception as ex:
                                print(f'worker done marker unreadable: {ex}', flush=True)
                                rc = -1
                        worker_cpu_total += w_cpu
                        log['worker_runs'].append({'step': step_idx, 'layers': f'{grp[0]}-{grp[-1]}',
                                                   'exit_code': rc, 'wall_s': round(w_wall, 2), 'cpu_s': round(w_cpu, 2)})
                        print(f'worker step {step_idx} layers {grp[0]}-{grp[-1]} exit {rc} out_exists {wout.exists()}', flush=True)
                        if rc != 0 or not wout.exists():
                            raise RuntimeError(f'worker failed step {step_idx} layers {grp[0]}-{grp[-1]} exit {rc}')
                        z = np.load(wout)
                        for i, e in enumerate(entries):
                            curs[f"{e['id']}_s{e['seed']}"] = z[f'cur{i}'].astype(np.float32)
                        z.close()
                        if a.size == 1024:
                            win.unlink(missing_ok=True); wout.unlink(missing_ok=True)
                    for e in entries:
                        t0 = time.perf_counter()
                        vel_t = rn.postprocess_np(gen, torch.from_numpy(curs[f"{e['id']}_s{e['seed']}"]), heads[f"{e['id']}_s{e['seed']}"], height=HW, width=HW)
                        st = states[f"{e['id']}_s{e['seed']}"]
                        st['lat'] = st['sch'].step(vel_t, torch.tensor(ts), st['lat'], return_dict=False)[0].float()
                        st['wall'] += time.perf_counter() - t0
            log['dit_wall_s'] = round(time.perf_counter() - t_dit0, 2)
            for e in entries:
                st = states[f"{e['id']}_s{e['seed']}"]
                final = st['lat'].numpy().astype(np.float32)
                name = f"{e['id']}_s{e['seed']}"
                expected = (1, 128, HW, HW)
                if final.shape != expected or final.dtype != np.float32 or not bool(np.isfinite(final).all()):
                    raise SystemExit(f'潜在の検査に失敗: {name} shape={final.shape} dtype={final.dtype} finite={bool(np.isfinite(final).all()) if final.size else False}')
                tmp_lat = lat_dir / f'{name}.tmp.npz'
                np.savez(tmp_lat, final_latents=final)
                os.replace(tmp_lat, lat_dir / f'{name}.npz')
                L_i = int(e['tokens']); nchunk_i = LAYER.plan(IMGTOK + L_i)[0]
                log['items'].append({'id': e['id'], 'seed': e['seed'], 'tokens': L_i, 't_chunks': int(nchunk_i),
                                     'finite': True, 'wall_4steps_s': round(st['wall'], 2), 'latents': str(lat_dir / f'{name}.npz')})
                print(f'dit saved {name} finite True', flush=True)
            if sessions is not None:
                rn.release_sessions(sessions)
            log['mem_final'] = rn.mem_snapshot()
        wall = time.perf_counter() - t_start; cpu = (cpu_seconds() - c_start) + worker_cpu_total
        # cpu は自プロセス (user+system) と worker が書いた CPU 時間の合計。worker_cpu_s は内訳。
        log['wall_s'] = round(wall, 2); log['cpu_s'] = round(cpu, 2); log['worker_cpu_s'] = round(worker_cpu_total, 2)
        tmp_log = out / 'dit_log.json.tmp'
        tmp_log.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp_log, out / 'dit_log.json')
        (out / f'dit_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'dit', 'run_id': a.run_id, 'cpu_s': round(cpu, 2), 'worker_cpu_s': round(worker_cpu_total, 2), 'wall_s': round(wall, 2)}), encoding='utf-8')
        print(f'dit done wall {wall:.2f}s cpu {cpu:.2f}s worker_cpu {worker_cpu_total:.2f}s', flush=True)
    except BaseException:
        try:
            wall = time.perf_counter() - t_start; cpu = cpu_seconds() - c_start
            (Path(a.out) / f'dit_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'dit', 'run_id': a.run_id, 'cpu_s': round(cpu, 2), 'worker_cpu_s': round(worker_cpu_total, 2), 'wall_s': round(wall, 2), 'failed': True}), encoding='utf-8')
        except Exception:
            pass
        raise

if __name__ == '__main__':
    main()
