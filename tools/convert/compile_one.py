"""DD 部品の初回コンパイル (NPU env, compile_fusion_rt=1 で 1 回実行し fconst を作る)。
入力は形の合った 0 で作る。P1 の参照ファイルは読まない。"""
from __future__ import annotations
import argparse, ctypes, os, sys, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part_dir')
    ap.add_argument('--onnx', default='replaced.onnx')
    ap.add_argument('--intra', type=int, default=2)
    ap.add_argument('--ryzen', required=True)
    a = ap.parse_args()
    import onnxruntime as ort
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    import npu_config as cfg
    part = cfg.ascii_alias(Path(a.part_dir))
    dd_dir = part / 'dd'
    custom_op = Path(a.ryzen) / 'deployment' / 'onnxruntime_providers_ryzenai.dll'
    os.add_dll_directory(str(custom_op.parent))
    ctypes.CDLL(str(custom_op))
    prev = os.getcwd()
    try:
        os.chdir(custom_op.parent)
        ort.register_execution_provider_library('RyzenAILightExecutionProvider', str(custom_op))
    finally:
        os.chdir(prev)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if a.intra:
        so.intra_op_num_threads = a.intra
    so.add_provider_for_devices([d for d in ort.get_ep_devices() if d.ep_name == 'RyzenAILightExecutionProvider'], {
        'dd_cache': (dd_dir / 'cache').as_posix(), 'onnx_custom_ops_const_key': '', 'compile_fusion_rt': '1'})
    so.register_custom_ops_library(str(custom_op))
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(dd_dir / a.onnx), sess_options=so)
    t_sess = time.perf_counter() - t0
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if i.type == 'tensor(int64)':
            feeds[i.name] = np.zeros(shape, np.int64)
        elif i.name == 'mark' or i.name.startswith('rope_cos'):
            feeds[i.name] = np.ones(shape, np.float32)
        else:
            feeds[i.name] = np.zeros(shape, np.float32)
    t1 = time.perf_counter()
    outs = sess.run(None, feeds)
    t_run = time.perf_counter() - t1
    finite = all(bool(np.isfinite(o).all()) for o in outs)
    print(f'{part.name}: session {t_sess:.2f}s run {t_run*1000:.1f}ms finite {finite}', flush=True)
    if not finite:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
