"""doctor.ps1 から NPU 環境で呼ぶ: A と T を 1 回ずつ小さな固定入力で回して有限か確かめる。"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import layer_var as lv
import layer_hd as lh
import prepost as rn

def probe(size: int):
    if size == 512:
        sa = lv.dd_session_var('attn_L0', 2)
        st = lv.dd_session_var('tile_nc_L0', 2)
        consts = lv.consts_for(np.zeros((1025, 64), np.float32), np.ones((1025, 64), np.float32), 1)
        x = np.zeros((1, 1216, 3072), np.float32)
        mod = np.zeros((1, 1, 3072), np.float32)
        feeds = {'x': x, 'shift_a': mod, 'scale_a': mod, **consts}
        feeds = {k: v for k, v in feeds.items() if k in {i.name for i in sa.get_inputs()}}
        att = sa.run(['att'], feeds)[0]
        ft = {'x': np.zeros((1, 256, 3072), np.float32), 'att': np.zeros((1, 256, 3072), np.float32),
              'gate_a': mod, 'shift_f': mod, 'scale_f': mod, 'gate_f': mod}
        out = st.run(['out'], ft)[0]
        return {'att_shape': list(att.shape), 'att_finite': bool(np.isfinite(att).all()),
                't_shape': list(out.shape), 't_finite': bool(np.isfinite(out).all())}
    sa = lh.dd_session_a1024('attn1024_L0', 2)
    st = lv.dd_session_var('tile_nc_L0', 2)
    consts = lh.consts_for(np.zeros((4097, 64), np.float32), np.ones((4097, 64), np.float32), 1)
    x = np.zeros((1, 4352, 3072), np.float32)
    mod = np.zeros((1, 1, 3072), np.float32)
    feeds = {'x': x, 'shift_a': mod, 'scale_a': mod, **consts}
    feeds = {k: v for k, v in feeds.items() if k in {i.name for i in sa.get_inputs()}}
    att = sa.run(['att'], feeds)[0]
    ft = {'x': np.zeros((1, 256, 3072), np.float32), 'att': np.zeros((1, 256, 3072), np.float32),
          'gate_a': mod, 'shift_f': mod, 'scale_f': mod, 'gate_f': mod}
    out = st.run(['out'], ft)[0]
    return {'att_shape': list(att.shape), 'att_finite': bool(np.isfinite(att).all()),
            't_shape': list(out.shape), 't_finite': bool(np.isfinite(out).all())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, choices=[512, 1024], default=512)
    a = ap.parse_args()
    import onnxruntime as ort
    r = probe(a.size)
    r['size'] = a.size
    r['providers'] = ort.get_available_providers()
    r['ep_devices'] = [str(d) for d in ort.get_ep_devices()]
    print(json.dumps(r, indent=1))
    if not (r['att_finite'] and r['t_finite']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
