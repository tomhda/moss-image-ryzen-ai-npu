"""doctor.ps1 から呼ぶ: 部品が manifest の大きさ・hash と一致するか確かめる。"""
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
    ap.add_argument('parts')
    ap.add_argument('--hash', action='store_true', help='大きさだけでなく SHA256 も確かめる')
    a = ap.parse_args()
    root = Path(a.parts)
    m = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    missing, sizediff, hashdiff = [], [], []
    for e in m['files']:
        f = root / e['path']
        if not f.exists():
            missing.append(e['path'])
        elif f.stat().st_size != e['size']:
            sizediff.append(e['path'])
        elif a.hash and sha256(f) != e['sha256']:
            hashdiff.append(e['path'])
    print(f"files={len(m['files'])} missing={missing[:5]} size_mismatch={sizediff[:5]} hash_mismatch={hashdiff[:5]}")
    print('github_commit', m.get('github_commit'))
    if missing or sizediff or hashdiff:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
