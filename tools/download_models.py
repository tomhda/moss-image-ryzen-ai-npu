"""setup.ps1 から呼ぶ: 部品と上流の必要ファイルだけを完全な commit hash で固定して落とす。
すでにあるファイルは manifest の hash を確かめて使う (部品)。"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', choices=['512', '1024', 'all'], required=True)
    ap.add_argument('--parts-dir', required=True)
    ap.add_argument('--moss-dir', required=True)
    ap.add_argument('--klein-dir', required=True)
    ap.add_argument('--hf-repo', required=True)
    ap.add_argument('--hf-revision', required=True)
    ap.add_argument('--moss-revision', required=True)
    ap.add_argument('--klein-revision', required=True)
    ap.add_argument('--skip-parts', action='store_true')
    ap.add_argument('--skip-models', action='store_true')
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    parts = Path(a.parts_dir)
    if not a.skip_parts:
        patterns = ['common/ffn/**', 'manifest.json', 'LICENSE']
        if a.size in ('512', 'all'):
            patterns.append('512/attention/**')
        if a.size in ('1024', 'all'):
            patterns.append('1024/attention/**')
        snapshot_download(repo_id=a.hf_repo, revision=a.hf_revision, allow_patterns=patterns,
                          local_dir=str(parts))
        # manifest と hash を確かめる
        mp = parts / 'manifest.json'
        if mp.exists():
            manifest = json.loads(mp.read_text(encoding='utf-8'))
            bad = []
            wanted_prefixes = tuple(p[:-3] for p in patterns if p.endswith('/**'))
            for e in manifest.get('files', []):
                if wanted_prefixes and not e['path'].startswith(wanted_prefixes):
                    continue
                f = parts / e['path']
                if not f.exists() or f.stat().st_size != e['size'] or sha256(f) != e['sha256']:
                    bad.append(e['path'])
            if bad:
                print(f'HASH MISMATCH {bad[:10]}', flush=True)
                raise SystemExit(1)
            print(f'parts verified size={a.size} files ok', flush=True)
    if not a.skip_models:
        moss_patterns = ['tokenizer/**', 'text_encoder/**', 'transformer/**', 'vae/**',
                         'pipeline_moss_image.py', 'scheduler/**', 'model_index.json', 'LICENSE', 'README.md']
        # すでにある場合は hash ではなく存在で判定し、落とし直さない (呼び出し側で既存フォルダを指す運用)
        moss_dir = Path(a.moss_dir)
        if (moss_dir / 'transformer').exists() and (moss_dir / 'text_encoder').exists():
            print(f'MOSS already present: {moss_dir} (skip download)', flush=True)
        else:
            snapshot_download(repo_id='OpenMOSS-Team/MOSS-Image-SFT', revision=a.moss_revision,
                              allow_patterns=moss_patterns, local_dir=str(moss_dir))
        klein_dir = Path(a.klein_dir)
        if (klein_dir / 'vae_decoder' / 'dynamic' / 'dd' / 'replaced.onnx').exists():
            print(f'klein VAE already present: {klein_dir} (skip download)', flush=True)
        else:
            snapshot_download(repo_id='amd/FLUX.2-klein-4B-amdnpu', revision=a.klein_revision,
                              allow_patterns=['vae_decoder/dynamic/dd/**'], local_dir=str(klein_dir))

if __name__ == '__main__':
    main()
