"""512^2 の 1 層 (NPU). A は replaced_var.onnx, T は FFN の塊 (replaced.onnx).
- A: x を 1216 行 (画像 1024 + 文字の枠 192) に 0 で広げて渡す. mark は本物の行 (1024+L) が 1. att は 1216 行で返る.
- T: 本物の行 (1024+L) だけを, 各 256 行の塊の 0-7 行と 160-167 行を捨て行にして流す."""
from __future__ import annotations
import numpy as np
import npu_config as cfg
import prepost as rn

M, IMG, TEXT_MAX = 1216, 1024, 192
CHUNK = 256
_BAD = set(range(0, 8)) | set(range(160, 168))
_REAL = [r for r in range(CHUNK) if r not in _BAD]
_plans: dict = {}
_bufs: dict = {}

def plan(rows: int):
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

def dd_session_var(part: str, intra=2, no_spin=False, no_arena=False):
    """part は attn_L{l} (A, replaced_var.onnx) か tile_nc_L{l} (T, replaced.onnx)."""
    if part.startswith('attn_'):
        layer = int(part[len('attn_L'):])
        return rn.dd_session_dir(cfg.attn512_dir(layer), 'replaced_var.onnx', intra, no_spin, no_arena)
    layer = int(part[len('tile_nc_L'):])
    return rn.dd_session_dir(cfg.ffn_dir(layer), 'replaced.onnx', intra, no_spin, no_arena)

def consts_for(rope_sin, rope_cos, text_len: int):
    assert 1 <= text_len <= TEXT_MAX
    st = np.zeros((TEXT_MAX, 64), np.float32); st[:text_len] = rope_sin[IMG:IMG + text_len]
    ct = np.ones((TEXT_MAX, 64), np.float32); ct[:text_len] = rope_cos[IMG:IMG + text_len]
    mark = np.zeros((1, M, 3072), np.float32); mark[0, :IMG + text_len] = 1.0
    return {'mark': mark, 'rope_cos_image': np.ascontiguousarray(rope_cos[:IMG], np.float32),
            'rope_sin_image': np.ascontiguousarray(rope_sin[:IMG], np.float32), 'rope_cos_text': ct, 'rope_sin_text': st}

def run_layer_var(sessions, layer, cur, mods, consts, want_x2=False):
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
