"""safetensors の重みを読み取り専用の写像のまま tensor にして、モデルに差し込む。

普通に読むと重みはプロセスの private メモリに写され、確保量（commit）に数えられる。
読み取り専用の写像ならファイルの頁のまま使えるので、確保量を増やさず、
触らない重み（文字エンコーダの画像側など）は物理メモリにも載らない。
"""
from __future__ import annotations

import json
import struct
import warnings
from pathlib import Path

import numpy as np
import torch

_DTYPES = {
    'BF16': (np.uint16, torch.bfloat16),
    'F16': (np.float16, None),
    'F32': (np.float32, None),
    'I64': (np.int64, None),
}


def open_safetensors(path) -> dict[str, torch.Tensor]:
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
    raw = np.memmap(path, dtype=np.uint8, mode='r')
    base = 8 + size
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        np_dtype, torch_view = _DTYPES[info['dtype']]
        start, end = info['data_offsets']
        array = raw[base + start:base + end]
        if (base + start) % np.dtype(np_dtype).itemsize:
            array = np.array(array)  # 境界がずれている重みだけ写す
        array = array.view(np_dtype).reshape(info['shape'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # 書き込めない配列の警告。重みは書き換えない
            tensor = torch.from_numpy(array)
        tensors[name] = tensor.view(torch_view) if torch_view is not None else tensor
    return tensors


def open_folder(folder) -> dict[str, torch.Tensor]:
    tensors = {}
    for path in sorted(Path(folder).glob('*.safetensors')):
        tensors.update(open_safetensors(path))
    return tensors


def assign(module: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> dict:
    """meta で作ったモデルに写像の tensor をそのまま差し込む。使わなかった名前と足りない名前を返す。"""
    result = module.load_state_dict(tensors, strict=False, assign=True)
    for p in module.parameters():
        p.requires_grad_(False)
    still_meta = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t.is_meta]
    return {'missing': list(result.missing_keys), 'unexpected': list(result.unexpected_keys), 'meta': still_meta}
