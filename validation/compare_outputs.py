"""比べる元と出力の照合: 最後の潜在 (npz 内の配列) と PNG 画素がビット単位で同じか。
ファイルの hash は別に記録する。使い方: compare_outputs.py <元フォルダ> <出力フォルダ> [--json out.json]"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('out')
    ap.add_argument('--json')
    a = ap.parse_args()
    ref, out = Path(a.ref), Path(a.out)
    rows = []
    ref_lat = sorted((ref / 'latents').glob('*.npz'))
    for rp in ref_lat:
        name = rp.name
        op = out / 'latents' / name
        row = {'name': name, 'lat_exists': op.exists()}
        if op.exists():
            rz = np.load(rp); oz = np.load(op)
            row['lat_keys'] = sorted(set(rz.files) | set(oz.files))
            row['lat_bit_equal'] = sorted(rz.files) == sorted(oz.files) and all(
                rz[k].shape == oz[k].shape and rz[k].dtype == oz[k].dtype and bool((rz[k] == oz[k]).all()) for k in rz.files)
            row['ref_npz_sha256'] = sha256(rp); row['out_npz_sha256'] = sha256(op)
            row['npz_file_equal'] = row['ref_npz_sha256'] == row['out_npz_sha256']
        png_name = name[:-4] + '.png' if name.endswith('.npz') else name + '.png'
        rp_png, op_png = ref / png_name, out / png_name
        row['png_exists'] = op_png.exists() and rp_png.exists()
        if row['png_exists']:
            ra = np.array(Image.open(rp_png)); oa = np.array(Image.open(op_png))
            row['png_bit_equal'] = ra.shape == oa.shape and ra.dtype == oa.dtype and bool((ra == oa).all())
            row['ref_png_sha256'] = sha256(rp_png); row['out_png_sha256'] = sha256(op_png)
            row['png_file_equal'] = row['ref_png_sha256'] == row['out_png_sha256']
        else:
            row['png_bit_equal'] = False
        row['ok'] = bool(row.get('lat_bit_equal')) and bool(row.get('png_bit_equal'))
        rows.append(row)
    ok = all(r['ok'] for r in rows) and len(rows) > 0
    rep = {'ref': str(ref), 'out': str(out), 'n': len(rows), 'ok': ok, 'items': rows}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    raise SystemExit(0 if ok else 1)

if __name__ == '__main__':
    main()
