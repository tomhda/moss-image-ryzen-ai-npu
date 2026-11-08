"""部品づくり用の層の重み。参照の出力ファイルは読まない。
形は解像度から決め、RoPE は幾何から作る。入力の答え合わせはしない (部品づくりと照合は分ける)。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mmap_weights import open_folder  # noqa: E402
from moss_model import LoRALinear, LORA_TARGETS, RANK, load_pipeline_module  # noqa: E402
from graph_build import DIM, FFN, HEADS, HEAD_DIM  # noqa: E402

DUMMY_TEXT_LEN = 41

def torch_block(mod, layer, weights, lora):
    block = mod._Block(DIM, FFN, HEADS)
    prefix = f'layers.{layer}.'
    block.load_state_dict({k[len(prefix):]: v.float() for k, v in weights.items() if k.startswith(prefix)}, strict=True)
    for name in LORA_TARGETS:
        wrapped = LoRALinear(block.proj[name], RANK, RANK)
        key = f'diffusion_model.layers.{layer}.proj.{name}'
        wrapped.lora_down.weight.data.copy_(lora[f'{key}.lora_down.weight'])
        wrapped.lora_up.weight.data.copy_(lora[f'{key}.lora_up.weight'])
        wrapped.alpha = float(lora[f'{key}.alpha'])
        block.proj[name] = wrapped
    return block.float().eval()

class CtxL:
    def __init__(self, layer: int, moss: Path, lora_path: Path, size: int):
        torch.set_grad_enabled(False)
        self.layer = layer
        self.size = size
        pm = load_pipeline_module(moss)
        self.pm = pm
        hw = 32 if size == 512 else 64
        self.height = self.width = hw
        self.image_tokens = hw * hw
        self.text_len = DUMMY_TEXT_LEN
        self.rows = self.image_tokens + self.text_len
        self.m_attn = 1216 if size == 512 else 4352
        weights = open_folder(moss / 'transformer')
        lora = load_file(str(lora_path))
        self.block = torch_block(pm, layer, weights, lora)
        rope = pm._RoPE(HEAD_DIM, 10000.0)(hw, hw, self.text_len, torch.device('cpu'))
        self.rope = rope
        self.sin = rope[0, :, :HEAD_DIM // 2]
        self.cos = rope[0, :, HEAD_DIM // 2:]

    def weight(self, name, lora='merged'):
        lin = self.block.proj[name]
        w = lin.base.weight.detach()
        if lora == 'merged':
            w = w + (lin.lora_up.weight @ lin.lora_down.weight) * (lin.alpha / lin.rank)
        return w.T.contiguous()
