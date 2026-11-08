"""設定は gen.ps1 が configs/local.json から読み、環境変数 MOSSNP_* で各段に渡す。"""
from __future__ import annotations
import os
from pathlib import Path

def req(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f'環境変数 {name} がありません（gen.ps1 / setup.ps1 経由で実行してください）')
    return Path(v)

def moss() -> Path:
    return req('MOSSNP_MOSS')

def ryzen_dll() -> Path:
    return req('MOSSNP_RYZEN') / 'deployment' / 'onnxruntime_providers_ryzenai.dll'

def parts_root() -> Path:
    return req('MOSSNP_PARTS')

def ffn_dir(layer: int) -> Path:
    return parts_root() / 'common' / 'ffn' / f'layer_{layer:02d}'

def attn512_dir(layer: int) -> Path:
    return parts_root() / '512' / 'attention' / f'layer_{layer:02d}'

def attn1024_dir(layer: int) -> Path:
    return parts_root() / '1024' / 'attention' / f'layer_{layer:02d}'

def klein_vae_dd() -> Path:
    return req('MOSSNP_KLEIN_VAE_DD')

def genai_sd() -> Path:
    return req('MOSSNP_GENAI_SD')
