"""MOSS 本体（利用者が取得したフォルダ）から pipeline モジュールと重みを読む。"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from torch import nn

LORA_TARGETS = ('q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down')
RANK = 128

class LoRALinear(nn.Module):
    """apply_lora.py と同じ: y = W x + (alpha / rank) * up(down(x))。LoRA は fp32 で、重みに混ぜない。"""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.rank, self.alpha = rank, alpha

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_up(self.lora_down(x.to(self.lora_down.weight.dtype)))
        return out + (delta * (self.alpha / self.rank)).to(out.dtype)

def load_pipeline_module(moss: Path):
    spec = importlib.util.spec_from_file_location('pipeline_moss_image', moss / 'pipeline_moss_image.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['pipeline_moss_image'] = module
    spec.loader.exec_module(module)
    return module

def materialize_rotary(model):
    """meta のまま残った回転位置の表（重みファイルに無い buffer）を CPU で作り直す。画像側は使わないので残す。"""
    fixed = []
    for name, module in model.named_modules():
        metas = [n for n, b in module.named_buffers(recurse=False) if b.is_meta]
        if not metas or name.startswith('visual'):
            continue
        import torch
        with torch.device('cpu'):
            fresh = type(module)(config=module.config)
        for buffer in metas:
            setattr(module, buffer, getattr(fresh, buffer))
        fixed.append(f'{name}: {metas}')
    return fixed
