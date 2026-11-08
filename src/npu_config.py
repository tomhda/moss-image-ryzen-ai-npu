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

def ascii_alias(p: Path) -> Path:
    """DD のネイティブ側は非 ASCII のパスを既存フォルダとして認識できないことがある。
    非 ASCII を含む場合は TEMP の下に ASCII 名のジャンクションを作ってそちらを返す (中身は複製しない)。"""
    try:
        str(p).encode('ascii')
        return p
    except UnicodeEncodeError:
        pass
    import hashlib, subprocess, tempfile
    alias = Path(tempfile.gettempdir()) / ('mossnpu-alias-' + hashlib.sha256(str(p).encode('utf-8')).hexdigest()[:10])
    marker = alias / '.mossnpu-target'
    if marker.exists() and marker.read_text(encoding='utf-8').strip() == str(p):
        return alias
    if alias.exists():
        # 別の対象を指すジャンクション/フォルダがある場合は作り直さない (安全側で失敗させる)
        raise SystemExit(f'ASCII alias の置き場が既にあります: {alias}')
    alias.mkdir(parents=True)
    alias.rmdir()
    # cmd の出力はコンソールのコードページ (日本語の Windows では cp932) なので oem で読む
    r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(p)], capture_output=True, text=True,
                       encoding='oem', errors='replace')
    if r.returncode != 0:
        raise SystemExit(f'mklink /J に失敗: {r.stdout} {r.stderr}')
    (alias / '.mossnpu-target').write_text(str(p), encoding='utf-8')
    return alias

def parts_root() -> Path:
    return ascii_alias(req('MOSSNP_PARTS'))

def ffn_dir(layer: int) -> Path:
    return parts_root() / 'common' / 'ffn' / f'layer_{layer:02d}'

def attn512_dir(layer: int) -> Path:
    return parts_root() / '512' / 'attention' / f'layer_{layer:02d}'

def attn1024_dir(layer: int) -> Path:
    return parts_root() / '1024' / 'attention' / f'layer_{layer:02d}'

def klein_vae_dd() -> Path:
    return ascii_alias(req('MOSSNP_KLEIN_VAE_DD'))

def genai_sd() -> Path:
    return req('MOSSNP_GENAI_SD')
