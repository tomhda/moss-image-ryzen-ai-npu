"""3 種の部品を一から作る: 作る (text env) -> 形付け -> 分割 -> flatten -> compile (NPU は 1 つずつ)。
512 の注意は var の形への書き換えまで。compile が済んだ部品から途中ファイルを消す。
使い方 (text env の python で):
  convert_all.py --moss <MOSS> --lora <lora.safetensors> --out <rebuild> --npu-python <python.exe> --ryzen <RyzenAI path> [--layers 0-41] [--kinds attn512,tile,attn1024]"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(cmd, env=None, log=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], env=env)
    dt = time.perf_counter() - t0
    print(f'  exit {r.returncode} {dt:.1f}s', flush=True)
    if log is not None:
        log.append({'cmd': [str(c) for c in cmd], 'exit': r.returncode, 'wall_s': round(dt, 1)})
    if r.returncode != 0:
        raise SystemExit(f'failed: {cmd[1] if len(cmd) > 1 else cmd} exit {r.returncode}')
    return dt

def parse_layers(s: str):
    if '-' in s and ',' not in s:
        lo, hi = s.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in s.split(',')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--moss', required=True)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--out', required=True, help='作り先 (例: rebuild)。下に parts/ と最終形 final/ を作る')
    ap.add_argument('--npu-python', required=True)
    ap.add_argument('--ryzen', required=True)
    ap.add_argument('--layers', default='0-41')
    ap.add_argument('--kinds', default='attn512,tile,attn1024')
    ap.add_argument('--keep-intermediate', action='store_true')
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    kinds = [k.strip() for k in a.kinds.split(',') if k.strip()]
    out = Path(a.out)
    parts = out / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    npu_py = Path(a.npu_python)
    onnx_utils = npu_py.parent / 'Scripts' / 'onnx_utils.exe'
    if not onnx_utils.exists():
        onnx_utils = npu_py.parent / 'onnx_utils.exe'
    npu_env = dict(os.environ)
    npu_env['PYTHONUTF8'] = '1'
    npu_env['PYTHONDONTWRITEBYTECODE'] = '1'
    prefix = str(npu_py.parent)
    npu_env['PATH'] = f"{prefix};{prefix}\\Library\\bin;{prefix}\\Scripts;" + npu_env.get('PATH', '')
    genai = Path(a.ryzen) / 'GenAI-SD'
    npu_env['RYZEN_AI_INSTALLATION_PATH'] = a.ryzen
    npu_env['DD_ROOT'] = str(genai / 'lib')
    npu_env['DD_PLUGINS_ROOT'] = str(genai / 'lib' / 'transaction' / 'stx')
    strat = HERE / 'strategies' / 'moss_block_dd.yaml'
    strat_nc = HERE / 'strategies' / 'moss_block_dd_nc.yaml'
    log = []
    t_all = time.perf_counter()
    jobs = []  # (kind, layer, src_part_name, final_rel)
    for kind in kinds:
        for l in layers:
            if kind == 'attn512':
                jobs.append((kind, l, f'attn_L{l}', f'512/attention/layer_{l:02d}'))
            elif kind == 'tile':
                jobs.append((kind, l, f'tile_L{l}', f'common/ffn/layer_{l:02d}'))
            else:
                jobs.append((kind, l, f'attn1024_L{l}', f'1024/attention/layer_{l:02d}'))
    for kind, l, src_name, final_rel in jobs:
        t0 = time.perf_counter()
        # 1. 作る (この text env の python)
        run([sys.executable, '-u', HERE / 'build_parts.py', kind, '--layer', l, '--moss', a.moss, '--lora', a.lora, '--out-parts', parts], log=log)
        part_dir = parts / src_name
        # 2. 形付け (NPU env)
        run([npu_py, '-u', HERE / 'shape_part.py', part_dir], env=npu_env, log=log)
        # 3. 分割 (onnx_utils)。T は cast_mul を外した nc の設定。
        yaml = strat_nc if kind == 'tile' else strat
        dd_out = part_dir / 'dd' / 'replaced.onnx'
        import shutil as _sh
        _sh.rmtree(part_dir / 'dd', ignore_errors=True)
        with open(part_dir / 'partition.log', 'w', encoding='utf-8') as lf:
            print(f'+ onnx_utils partition {src_name}', flush=True)
            t1 = time.perf_counter()
            r = subprocess.run([str(onnx_utils), 'partition', str(part_dir / 'shaped.onnx'), str(dd_out), str(yaml),
                                '--attributes', 'whitebox_checker_level=2', '--save-as-external', '--force'],
                               env=npu_env, stdout=lf, stderr=subprocess.STDOUT)
            print(f'  exit {r.returncode} {time.perf_counter()-t1:.1f}s', flush=True)
            log.append({'cmd': ['onnx_utils', 'partition', src_name], 'exit': r.returncode, 'wall_s': round(time.perf_counter() - t1, 1)})
            if r.returncode != 0:
                raise SystemExit(f'partition failed {src_name}')
        # 4. flatten
        run([sys.executable, '-u', HERE / 'flatten_dd.py', part_dir], log=log)
        # tile は最終名を tile_nc に合わせる (中身は同じ。置き先で分かるように記録)
        # 5. compile (NPU, 1 つずつ)。DD セッションの終了処理でアクセス違反になることがあるため、
        # 成功は compile_one の finite True の出力で判定する (研究段階でも同じ扱い)。
        print(f'+ compile_one {src_name}', flush=True)
        t1 = time.perf_counter()
        r = subprocess.run([str(npu_py), '-u', str(HERE / 'compile_one.py'), str(part_dir), '--ryzen', a.ryzen],
                           env=npu_env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        sys.stdout.write(r.stdout[-2000:] if r.stdout else '')
        if r.stderr:
            sys.stdout.write(r.stderr[-1000:])
        ok_compile = 'finite True' in (r.stdout or '')
        log.append({'cmd': ['compile_one', src_name], 'exit': r.returncode, 'wall_s': round(time.perf_counter() - t1, 1), 'finite_true': ok_compile})
        print(f'  compile exit {r.returncode} finite_true {ok_compile}', flush=True)
        if not ok_compile:
            raise SystemExit(f'compile failed {src_name}')
        # 6. 512 の注意は var へ書き換え
        if kind == 'attn512':
            run([sys.executable, '-u', HERE / 'make_var_attn.py', part_dir], log=log)
        # 最終形へ dd の中身を移す (ハードリンクではなく複製。rebuild 内の作業のため)
        final_dir = out / 'final' / Path(final_rel)
        final_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        src_dd = part_dir / 'dd'
        for f in src_dd.rglob('*'):
            if f.is_file():
                dst = final_dir / f.relative_to(src_dd)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
        # 途中ファイルを消す
        if not a.keep_intermediate:
            for name in ('model.onnx', 'model.onnx.data', 'shaped.onnx', 'shaped.onnx.data'):
                (part_dir / name).unlink(missing_ok=True)
        print(f'{kind} L{l} done {time.perf_counter()-t0:.1f}s -> {final_rel}', flush=True)
        (out / 'convert_log.json').write_text(json.dumps({'jobs': log, 'wall_s': round(time.perf_counter() - t_all, 1)}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'all done {time.perf_counter()-t_all:.1f}s jobs {len(jobs)}', flush=True)

if __name__ == '__main__':
    main()
