"""部品の ONNX に形の情報を付ける (NPU env)。"""
from __future__ import annotations
import sys
from pathlib import Path
import onnx
import ryzenai_onnx_utils.preprocess

def shape_one(part_dir: Path):
    src = part_dir / 'model.onnx'
    dst = part_dir / 'shaped.onnx'
    model = onnx.load_model(src, load_external_data=True)
    shaped = ryzenai_onnx_utils.preprocess.infer_symbolic_shapes(model)
    have = {v.name for v in shaped.graph.value_info}
    shaped.graph.value_info.extend([v for v in shaped.graph.input if v.name not in have])
    known = {v.name for v in list(shaped.graph.value_info) + list(shaped.graph.output) + list(shaped.graph.input)}
    missing = [o for n in shaped.graph.node for o in n.output if o not in known]
    (dst.parent / (dst.name + '.data')).unlink(missing_ok=True)
    onnx.save_model(shaped, dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=dst.name + '.data', size_threshold=1024)
    print(part_dir.name, f'value_info {len(shaped.graph.value_info)} missing {len(missing)}', flush=True)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        shape_one(Path(p))
