"""1024^2 の 1 層 (NPU). A は attn1024 (replaced.onnx), T は FFN の塊.
- A: x を 4352 行 (画像 4096 + 文字の枠 256) に 0 で広げて渡す. mark は本物の行 (4096+L) が 1. att は 4352 行で返る.
- T: 本物の行 (4096+L) だけを, 各 256 行の塊の 0-7 行と 160-167 行を捨て行にして流す."""
from __future__ import annotations
import numpy as np
import npu_config as cfg
import prepost as rn

PARTS_HD = None
M, IMG, TEXT_FRAME = 4352, 4096, 256
CHUNK = 256
_BAD = set(range(0, 8)) | set(range(160, 168))
_REAL = [r for r in range(CHUNK) if r not in _BAD]
_plans: dict = {}
_bufs: dict = {}
_dd_registered = [False]

def plan(rows: int):
    """rows 行を塊に並べる段取り（layer_var.plan と同じ）。"""
    if rows in _plans:
        return _plans[rows]
    nchunk = -(-rows // len(_REAL))
    slot = np.array([c * CHUNK + r for c in range(nchunk) for r in _REAL])[:rows]
    pos = np.full(nchunk * CHUNK, -1, np.int64); pos[slot] = np.arange(rows)
    src = np.empty(nchunk * CHUNK, np.int64); last = 0
    for p in range(nchunk * CHUNK):
        if pos[p] >= 0:
            last = pos[p]
        src[p] = pos[p] if pos[p] >= 0 else min(last + (1 if p % CHUNK < 8 else 0), rows - 1)
    spans = []
    for c in range(nchunk):
        sp = []
        for lo, hi in ((8, 160), (168, CHUNK)):
            o0 = int(pos[c * CHUNK + lo])
            if o0 < 0:
                continue
            n = min(hi - lo, rows - o0)
            sp.append((lo, lo + n, o0))
        spans.append(sp)
    _plans[rows] = (nchunk, src, spans)
    return _plans[rows]


def _buf(name, shape, dtype):
    b = _bufs.get(name)
    if b is None or b.shape != shape or b.dtype != dtype:
        b = _bufs[name] = np.empty(shape, dtype)
    return b


def dd_session_a1024(part: str, intra=2, no_spin=False, no_arena=False):
    """part は attn1024_L{l}. compile_fusion_rt は 0."""
    layer = int(part[len('attn1024_L'):])
    return rn.dd_session_dir(cfg.attn1024_dir(layer), 'replaced.onnx', intra, no_spin, no_arena)


def consts_for(rope_sin, rope_cos, text_len: int):
    """A の定数。rope_sin/cos は [4096+L, 64]（rope_mod(64, 64, L) の sin・cos の半分）。"""
    assert 1 <= text_len <= TEXT_FRAME
    st = np.zeros((TEXT_FRAME, 64), np.float32); st[:text_len] = rope_sin[IMG:IMG + text_len]
    ct = np.ones((TEXT_FRAME, 64), np.float32); ct[:text_len] = rope_cos[IMG:IMG + text_len]
    mark = np.zeros((1, M, 3072), np.float32); mark[0, :IMG + text_len] = 1.0
    return {'mark': mark, 'rope_cos_image': np.ascontiguousarray(rope_cos[:IMG], np.float32),
            'rope_sin_image': np.ascontiguousarray(rope_sin[:IMG], np.float32), 'rope_cos_text': ct, 'rope_sin_text': st}


def run_layer_hd(sessions, layer, cur, mods, consts, want_x2=False):
    sa, st = sessions[f'{layer}:A'], sessions[f'{layer}:T']
    rows = cur.shape[1]
    assert IMG < rows <= M, rows
    xa = _buf('xa', (1, M, cur.shape[2]), cur.dtype)
    xa[:, :rows] = cur; xa[:, rows:] = 0.0
    fa = {'x': xa, 'shift_a': mods['shift_a'], 'scale_a': mods['scale_a'], **consts}
    fa = {k: v for k, v in fa.items() if k in {i.name for i in sa.get_inputs()}}
    oa, dt_a = rn.run_timed(sa, ['att'], fa)
    nchunk, src, spans = plan(rows)
    shape = (1, nchunk * CHUNK, cur.shape[2])
    xs = np.take(cur, src, axis=1, out=_buf(f'x{nchunk}', shape, cur.dtype))
    ats = np.take(oa['att'], src, axis=1, out=_buf(f'att{nchunk}', shape, cur.dtype))
    out = np.empty_like(cur)
    x2 = np.empty_like(cur) if want_x2 else None
    names = ['out', 'x2'] if want_x2 else ['out']
    dt_t = 0.0
    for c in range(nchunk):
        ft = {'x': xs[:, c * CHUNK:(c + 1) * CHUNK], 'att': ats[:, c * CHUNK:(c + 1) * CHUNK],
              'gate_a': mods['gate_a'], 'shift_f': mods['shift_f'], 'scale_f': mods['scale_f'], 'gate_f': mods['gate_f']}
        oc, dt = rn.run_timed(st, names, ft)
        dt_t += dt
        for lo, hi, o0 in spans[c]:
            out[:, o0:o0 + hi - lo] = oc['out'][:, lo:hi]
            if want_x2:
                x2[:, o0:o0 + hi - lo] = oc['x2'][:, lo:hi]
    return out, x2, dt_a + dt_t
