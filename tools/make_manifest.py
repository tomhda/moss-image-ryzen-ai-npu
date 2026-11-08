"""配る部品の manifest.json を作る。各ファイルの相対パス・大きさ・SHA256、
上流の版、Ryzen AI の版と wheel の版、部品の形、対応する GitHub の commit を書く。"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts-root', required=True, help='HF に並べた部品のルート (common/, 512/, 1024/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--github-commit', required=True)
    ap.add_argument('--moss-repo', default='OpenMOSS-Team/MOSS-Image-SFT')
    ap.add_argument('--moss-revision', required=True)
    ap.add_argument('--lora-repo', default='akhaliq/MOSS-Image-SFT-4-steps')
    ap.add_argument('--lora-revision', required=True)
    ap.add_argument('--klein-repo', default='amd/FLUX.2-klein-4B-amdnpu')
    ap.add_argument('--klein-revision', required=True)
    ap.add_argument('--hf-repo', default='Tomoali/MOSS-Image-SFT-4-steps-ryzen-ai-npu')
    ap.add_argument('--sdk-version', default='1.8.0')
    a = ap.parse_args()
    root = Path(a.parts_root)
    wheels = {}
    try:
        import importlib.metadata as md
        for dist in ('ryzenai_onnx_utils', 'ryzenai_dynamic_dispatch', 'onnxruntime_providers_ryzenai', 'onnxruntime_vitisai'):
            try:
                wheels[dist] = md.version(dist)
            except Exception:
                wheels[dist] = None
    except Exception:
        wheels = {}
    files = []
    total = 0
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name != 'manifest.json':
            rel = p.relative_to(root).as_posix()
            size = p.stat().st_size
            total += size
            files.append({'path': rel, 'size': size, 'sha256': sha256(p)})
    manifest = {
        'github_commit': a.github_commit,
        'hf_repo': a.hf_repo,
        'upstream': {
            'moss': {'repo': a.moss_repo, 'revision': a.moss_revision},
            'lora': {'repo': a.lora_repo, 'revision': a.lora_revision},
            'klein_vae': {'repo': a.klein_repo, 'revision': a.klein_revision},
        },
        'ryzen_ai': {'sdk_version': a.sdk_version, 'wheels': wheels},
        'parts': {
            'common/ffn': {'shape': 'FFN chunk, 256 rows, width 3072', 'layers': 42},
            '512/attention': {'shape': 'sequence 1216 (image 1024 + text frame 192)', 'text_frame': 192, 'layers': 42},
            '1024/attention': {'shape': 'sequence 4352 (image 4096 + text frame 256)', 'text_frame': 256, 'layers': 42},
        },
        'total_size': total,
        'files': files,
    }
    Path(a.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'files {len(files)} total {total} -> {a.out}')

if __name__ == '__main__':
    main()
