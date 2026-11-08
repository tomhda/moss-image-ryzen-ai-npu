"""部品の ONNX を作る (text env)。512 の注意 attn_L{l}、FFN の塊 tile_L{l}、1024 の注意 attn1024_L{l}。
重みは層 L の merged (fp32 で混ぜる)。答え合わせの入力は作らない。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_build import DIM, FFN, HEADS, HEAD_DIM, GraphBuilder, std_rmsnorm  # noqa: E402
from ctx import CtxL  # noqa: E402

FFN_PAD = 9216
HALF_DIM = HEAD_DIM // 2
PERM_HEAD = [j // 2 + (HALF_DIM if j % 2 else 0) for j in range(HEAD_DIM)]
PERM = torch.tensor([h * HEAD_DIM + p for h in range(HEADS) for p in PERM_HEAD])
BLOCK_MODS = ('shift_a', 'scale_a', 'gate_a', 'shift_f', 'scale_f', 'gate_f')

def save_part(out_dir: Path, name, g, inputs, outputs, note, **extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    f32 = TensorProto.FLOAT
    def vi(n, s):
        return helper.make_tensor_value_info(n, TensorProto.INT64 if n == 'position_ids' else f32, s)
    graph = helper.make_graph(g.nodes, name, [vi(n, s) for n, s in inputs], [vi(n, s) for n, s in outputs], g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('com.microsoft', 1)])
    model.ir_version = 8
    path = out_dir / 'model.onnx'
    (out_dir / 'model.onnx.data').unlink(missing_ok=True)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location='model.onnx.data', size_threshold=1024)
    meta = {'name': name, 'note': note, 'inputs': {n: s for n, s in inputs}, 'outputs': {n: s for n, s in outputs},
            'nodes': len(g.nodes), 'op_types': sorted({n.op_type for n in g.nodes}), **extra}
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(name, meta['nodes'], meta['op_types'], flush=True)
    return meta

def _modulate_std(g, normed, scale, shift, base):
    one_plus = g.op('Add', [scale, g.const(f'{base}.one', np.array(1.0, np.float32))], f'{base}/one_plus')
    return g.op('Add', [g.op('Mul', [normed, one_plus], f'{base}/mul'), shift], f'{base}/add')

def _pad_rows(g, x, pad, base):
    pads = g.const(f'{base}.pads', np.array([0, 0, 0, 0, pad, 0], np.int64))
    return g.op('Pad', [x, pads], f'{base}/Pad', mode='constant')

def _slice_rows(g, x, end, base):
    starts = g.const(f'{base}.starts', np.array([0], np.int64))
    ends = g.const(f'{base}.ends', np.array([end], np.int64))
    axes = g.const(f'{base}.axes', np.array([1], np.int64))
    return g.op('Slice', [x, starts, ends, axes], f'{base}/Slice')

def attn_front_512(g, ctx, x_name, shift_a, scale_a, positions, eye, prefix=''):
    rows, m = ctx.rows, ctx.m_attn
    y = _modulate_std(g, std_rmsnorm(g, x_name, ctx.block.norm1.weight, f'{prefix}norm1'), scale_a, shift_a, f'{prefix}mod_a')
    y_pad = _pad_rows(g, y, m - rows, f'{prefix}y_pad')
    qk = {}
    for nm, norm in (('q', ctx.block.q_norm), ('k', ctx.block.k_norm)):
        w = ctx.weight(nm, 'merged')[:, PERM].contiguous()
        p = g.op('MatMul', [y_pad, g.const(f'{prefix}proj.{nm}.w', w.numpy())], f'{prefix}proj.{nm}/MatMul')
        heads = g.op('Reshape', [p, g.const(f'{prefix}{nm}.to_heads', np.array([1, m, HEADS, HEAD_DIM], np.int64))], f'{prefix}{nm}_norm/to_heads')
        normed = g.rmsnorm(heads, norm.weight.detach()[PERM_HEAD], f'{prefix}{nm}_norm')
        qk[nm] = g.op('Reshape', [normed, g.const(f'{prefix}{nm}.to_flat', np.array([1, m, DIM], np.int64))], f'{prefix}{nm}_norm/to_flat')
    for nm in ('rope_cos', 'rope_sin'):
        g.nodes.append(helper.make_node('Concat', [f'{nm}_image', f'{nm}_text'], [g.name(f'{prefix}{nm}')], name=f'{prefix}{nm}/Concat_{g.count}', axis=0))
        g.nodes[-1].output[0] = f'{prefix}rope_cos' if nm == 'rope_cos' else f'{prefix}rope_sin'
    rope_cos, rope_sin = f'{prefix}rope_cos', f'{prefix}rope_sin'
    v_real = g.op('MatMul', [y_pad, g.const(f'{prefix}proj.v.w', ctx.weight('v', 'merged').numpy())], f'{prefix}proj.v/MatMul')
    v_mark = g.op('MatMul', ['mark', eye], f'{prefix}mark/MatMul')
    att_out = {}
    for tag, v_in in (('y0', v_real), ('a', v_mark)):
        rot = {}
        for nm in ('q', 'k'):
            rot[nm] = g.op('RotaryEmbedding', [qk[nm], positions, rope_cos, rope_sin], f'{prefix}{tag}/{nm}_rope',
                           domain='com.microsoft', interleaved=1, num_heads=HEADS, rotary_embedding_dim=HEAD_DIM)
        g.op('MultiHeadAttention', [rot['q'], rot['k'], v_in], f'{prefix}{tag}/attn', domain='com.microsoft', num_heads=HEADS)
        att_out[tag] = g.nodes[-1].output[0]
    return _slice_rows(g, g.op('Div', [att_out['y0'], att_out['a']], f'{prefix}att/div'), rows, f'{prefix}att')

def attn_front_var(g, ctx, x_name, shift_a, scale_a, positions, eye, prefix=''):
    m = ctx.m_attn
    y = _modulate_std(g, std_rmsnorm(g, x_name, ctx.block.norm1.weight, f'{prefix}norm1'), scale_a, shift_a, f'{prefix}mod_a')
    y = g.op('Mul', [y, 'mark'], f'{prefix}y_mask')
    qk = {}
    for nm, norm in (('q', ctx.block.q_norm), ('k', ctx.block.k_norm)):
        w = ctx.weight(nm, 'merged')[:, PERM].contiguous()
        p = g.op('MatMul', [y, g.const(f'{prefix}proj.{nm}.w', w.numpy())], f'{prefix}proj.{nm}/MatMul')
        heads = g.op('Reshape', [p, g.const(f'{prefix}{nm}.to_heads', np.array([1, m, HEADS, HEAD_DIM], np.int64))], f'{prefix}{nm}_norm/to_heads')
        normed = g.rmsnorm(heads, norm.weight.detach()[PERM_HEAD], f'{prefix}{nm}_norm')
        qk[nm] = g.op('Reshape', [normed, g.const(f'{prefix}{nm}.to_flat', np.array([1, m, DIM], np.int64))], f'{prefix}{nm}_norm/to_flat')
    for nm in ('rope_cos', 'rope_sin'):
        g.nodes.append(helper.make_node('Concat', [f'{nm}_image', f'{nm}_text'], [g.name(f'{prefix}{nm}')], name=f'{prefix}{nm}/Concat_{g.count}', axis=0))
        g.nodes[-1].output[0] = f'{prefix}rope_cos' if nm == 'rope_cos' else f'{prefix}rope_sin'
    rope_cos, rope_sin = f'{prefix}rope_cos', f'{prefix}rope_sin'
    v_real = g.op('MatMul', [y, g.const(f'{prefix}proj.v.w', ctx.weight('v', 'merged').numpy())], f'{prefix}proj.v/MatMul')
    v_mark = g.op('MatMul', ['mark', eye], f'{prefix}mark/MatMul')
    att_out = {}
    for tag, v_in in (('y0', v_real), ('a', v_mark)):
        rot = {}
        for nm in ('q', 'k'):
            rot[nm] = g.op('RotaryEmbedding', [qk[nm], positions, rope_cos, rope_sin], f'{prefix}{tag}/{nm}_rope',
                           domain='com.microsoft', interleaved=1, num_heads=HEADS, rotary_embedding_dim=HEAD_DIM)
        g.op('MultiHeadAttention', [rot['q'], rot['k'], v_in], f'{prefix}{tag}/attn', domain='com.microsoft', num_heads=HEADS)
        att_out[tag] = g.nodes[-1].output[0]
    return g.op('Div', [att_out['y0'], att_out['a']], f'{prefix}att/div')

def ffn_weights(ctx):
    w_ug = ctx.weight('ffn_up_gate', 'merged')
    w_hidden, w_gate = torch.zeros(DIM, FFN_PAD), torch.zeros(DIM, FFN_PAD)
    w_hidden[:, :FFN], w_gate[:, :FFN] = w_ug[:, :FFN], w_ug[:, FFN:]
    w_down = torch.zeros(FFN_PAD, DIM)
    w_down[:FFN] = ctx.weight('ffn_down', 'merged')
    return w_hidden, w_gate, w_down

def situ_ffn_chunk(g, chunk, c_hidden, c_gate, c_down, c4, c25, prefix):
    hidden = g.op('MatMul', [chunk, c_hidden], f'{prefix}ffn_hidden/MatMul')
    gate = g.op('MatMul', [chunk, c_gate], f'{prefix}ffn_gate/MatMul')
    t = g.op('Mul', [c4, g.op('Tanh', [g.op('Div', [gate, c4], f'{prefix}situ/gate_div')], f'{prefix}situ/gate_tanh')], f'{prefix}situ/gate_mul4')
    t = g.op('Mul', [t, g.op('Sigmoid', [gate], f'{prefix}situ/gate_sigmoid')], f'{prefix}situ/gate_mul_sig')
    h = g.op('Mul', [c25, g.op('Tanh', [g.op('Div', [hidden, c25], f'{prefix}situ/hidden_div')], f'{prefix}situ/hidden_tanh')], f'{prefix}situ/hidden_mul25')
    act = g.op('Mul', [t, h], f'{prefix}situ/out')
    return g.op('MatMul', [act, c_down], f'{prefix}ffn_down/MatMul')

def shared_consts(g, ctx):
    positions = g.const('position_ids', np.arange(ctx.m_attn, dtype=np.int64)[None])
    eye = g.const('eye', np.eye(DIM, dtype=np.float32))
    return positions, eye

def rope_io_shapes(ctx):
    return [('rope_cos_image', [ctx.image_tokens, HALF_DIM]), ('rope_sin_image', [ctx.image_tokens, HALF_DIM]),
            ('rope_cos_text', [ctx.m_attn - ctx.image_tokens, HALF_DIM]), ('rope_sin_text', [ctx.m_attn - ctx.image_tokens, HALF_DIM])]

def build_attn512(ctx, out_root: Path):
    g = GraphBuilder()
    positions, eye = shared_consts(g, ctx)
    att = attn_front_512(g, ctx, 'x', 'shift_a', 'scale_a', positions, eye, '')
    for n in g.nodes:
        if att in n.output:
            n.output[list(n.output).index(att)] = 'att'
            break
    rows = ctx.rows
    inputs = [('x', [1, rows, DIM]), ('shift_a', [1, 1, DIM]), ('scale_a', [1, 1, DIM]), ('mark', [1, ctx.m_attn, DIM])] + rope_io_shapes(ctx)
    save_part(out_root / f'attn_L{ctx.layer}', f'attn_L{ctx.layer}', g, inputs, [('att', [1, rows, DIM])],
              f'512 attention layer {ctx.layer}', layer=ctx.layer, rows=rows, image_tokens=ctx.image_tokens)

def build_attn1024(ctx, out_root: Path):
    g = GraphBuilder()
    positions, eye = shared_consts(g, ctx)
    att = attn_front_var(g, ctx, 'x', 'shift_a', 'scale_a', positions, eye, '')
    for n in g.nodes:
        if att in n.output:
            n.output[list(n.output).index(att)] = 'att'
            break
    m = ctx.m_attn
    inputs = [('x', [1, m, DIM]), ('shift_a', [1, 1, DIM]), ('scale_a', [1, 1, DIM]), ('mark', [1, m, DIM])] + rope_io_shapes(ctx)
    save_part(out_root / f'attn1024_L{ctx.layer}', f'attn1024_L{ctx.layer}', g, inputs, [('att', [1, m, DIM])],
              f'1024 attention layer {ctx.layer}', layer=ctx.layer, rows=ctx.rows, image_tokens=ctx.image_tokens)

def build_tile(ctx, out_root: Path):
    g = GraphBuilder()
    w_o = g.const('proj.o.w', ctx.weight('o', 'merged').numpy())
    o = g.op('MatMul', ['att', w_o], 'o/MatMul')
    x2 = g.op('Add', ['x', g.op('Mul', [o, 'gate_a'], 'gate_a/mul')], 'residual_a/add')
    g.nodes[-1].output[0] = 'x2'
    y2 = _modulate_std(g, std_rmsnorm(g, 'x2', ctx.block.norm2.weight, 'norm2'), 'scale_f', 'shift_f', 'mod_f')
    w_hidden, w_gate, w_down = ffn_weights(ctx)
    c_hidden = g.const('ffn_hidden.w', w_hidden.numpy())
    c_gate = g.const('ffn_gate.w', w_gate.numpy())
    c_down = g.const('ffn_down.w', w_down.numpy())
    c4, c25 = g.const('situ.4', np.array(4.0, np.float32)), g.const('situ.25', np.array(25.0, np.float32))
    ffn = situ_ffn_chunk(g, y2, c_hidden, c_gate, c_down, c4, c25, '')
    out = g.op('Add', ['x2', g.op('Mul', [ffn, 'gate_f'], 'gate_f/mul')], 'residual_f/add')
    g.nodes[-1].output[0] = 'out'
    inputs = [('x', [1, 256, DIM]), ('att', [1, 256, DIM]), ('gate_a', [1, 1, DIM]), ('shift_f', [1, 1, DIM]),
              ('scale_f', [1, 1, DIM]), ('gate_f', [1, 1, DIM])]
    save_part(out_root / f'tile_L{ctx.layer}', f'tile_L{ctx.layer}', g, inputs, [('out', [1, 256, DIM]), ('x2', [1, 256, DIM])],
              f'FFN chunk layer {ctx.layer}', layer=ctx.layer, rows=ctx.rows, image_tokens=ctx.image_tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['attn512', 'attn1024', 'tile'])
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--moss', required=True)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--out-parts', required=True, help='model.onnx を置く部品フォルダの親')
    a = ap.parse_args()
    size = 1024 if a.kind == 'attn1024' else 512
    ctx = CtxL(a.layer, Path(a.moss), Path(a.lora), size)
    out_root = Path(a.out_parts)
    if a.kind == 'attn512':
        build_attn512(ctx, out_root)
    elif a.kind == 'attn1024':
        build_attn1024(ctx, out_root)
    else:
        build_tile(ctx, out_root)

if __name__ == '__main__':
    main()
