"""doctor.ps1 から NPU 環境で呼ぶ: A と T を 1 回ずつ小さな固定入力で回して有限か確かめる。"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import layer_var as lv

def probe(size: int):
    """生成の worker と同じ作り方・回し方で、層 0（A と T）を小さな固定入力で 1 回通す。"""
    import stage_dit as sd
    import prepost as rn
    import layer_hd as lh
    lay = lv if size == 512 else lh
    run_layer = lv.run_layer_var if size == 512 else lh.run_layer_hd
    opts = argparse.Namespace(intra=2, no_spin=False, no_arena=False, size=size)
    sessions, _, create_s, fail = sd.create_sessions_p5([0], opts)
    if sessions is None:
        return {'finite': False, 'error': f'session create stopped at layer {fail} (low memory)'}
    text_len = 8
    rows = lay.IMG + text_len
    cur = (np.random.default_rng(0).standard_normal((1, rows, 3072)) * 0.1).astype(np.float32)
    mods = {nm: np.zeros((1, 1, 3072), np.float32) for nm in rn.BLOCK_MODS}
    consts = lay.consts_for(np.zeros((rows, 64), np.float32), np.ones((rows, 64), np.float32), text_len)
    out, _, _ = run_layer(sessions, 0, cur, mods, consts)
    return {'layer': 0, 'rows': rows, 'out_shape': list(out.shape), 'finite': bool(np.isfinite(out).all()),
            'create_s': round(create_s, 2)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, choices=[512, 1024], default=512)
    a = ap.parse_args()
    import onnxruntime as ort
    providers = ort.get_available_providers()
    r = probe(a.size)
    r['size'] = a.size
    r['providers'] = providers
    ok = bool(r['finite'])
    r['ok'] = ok
    text = json.dumps(r, indent=1)
    print(text, flush=True)
    # DD セッションの終了処理で落ちることがあるため、結果を出してから OS 終了する
    sys.stdout.flush()
    import os as _os
    _os._exit(0 if ok else 1)

if __name__ == '__main__':
    main()
