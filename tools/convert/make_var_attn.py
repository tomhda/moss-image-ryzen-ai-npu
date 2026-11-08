"""512 の注意 A を文字の長さを変えられる形に書き換える。NPU の本体とキャッシュはそのまま。
CPU 側だけ: 入力 x を 1065 行 -> 1216 行、Pad -> Mul(変調の後, mark)、最後の Slice を外す。"""
import sys
from pathlib import Path
import onnx
from onnx import helper

M = 1216

def convert(part_dir: Path) -> Path:
    src = part_dir / 'dd' / 'replaced.onnx'
    m = onnx.load(str(src), load_external_data=False)
    g = m.graph
    pads = [n for n in g.node if n.op_type == 'Pad']
    slices = [n for n in g.node if n.op_type == 'Slice']
    assert len(pads) == 1 and len(slices) == 1, (len(pads), len(slices))
    pad, sl = pads[0], slices[0]
    mul = helper.make_node('Mul', [pad.input[0], 'mark'], list(pad.output), name='y_mask/Mul')
    g.node.insert(list(g.node).index(pad), mul)
    g.node.remove(pad)
    div_out = sl.input[0]
    for n in g.node:
        for i, o in enumerate(n.output):
            if o == div_out:
                n.output[i] = 'att'
    g.node.remove(sl)
    for coll in (g.input, g.output, g.value_info):
        for v in coll:
            dims = v.type.tensor_type.shape.dim
            if len(dims) == 3 and dims[1].dim_value == 1065:
                dims[1].dim_value = M
    used = {i for n in g.node for i in n.input}
    for init in list(g.initializer):
        if init.name not in used and (init.name.startswith('y_pad.pads') or init.name.startswith('att.')):
            g.initializer.remove(init)
    dst = src.with_name('replaced_var.onnx')
    onnx.save(m, str(dst))
    return dst

if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(convert(Path(p)), flush=True)
