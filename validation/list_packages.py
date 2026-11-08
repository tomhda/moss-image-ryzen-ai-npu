import sys, importlib.metadata as m
print(sys.version)
for d in ('torch', 'numpy', 'onnxruntime-vitisai', 'ryzenai_onnx_utils', 'transformers', 'diffusers', 'safetensors', 'onnx'):
    try:
        print(f'{d}=={m.version(d)}')
    except Exception:
        print(f'{d}==MISSING')
