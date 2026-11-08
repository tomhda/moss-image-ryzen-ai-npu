"""partition の入れ子出力 (dd/replaced.onnx という名前のフォルダ) を dd 直下へ直す。"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

def flatten(part_dir: Path) -> str:
    dd = part_dir / 'dd'
    inner = dd / 'replaced.onnx'
    if inner.is_file():
        return f'{part_dir.name}: already flat'
    if not inner.is_dir():
        return f'{part_dir.name}: MISSING {inner}'
    tmp = dd / '_replaced_tmp.onnx'
    shutil.move(str(inner / 'replaced.onnx'), str(tmp))
    if (inner / 'replaced.onnx.data').exists():
        shutil.move(str(inner / 'replaced.onnx.data'), str(dd / 'replaced.onnx.data'))
    if (inner / 'cache').exists():
        if (dd / 'cache').exists():
            shutil.rmtree(dd / 'cache')
        shutil.move(str(inner / 'cache'), str(dd / 'cache'))
    for f in list(inner.iterdir()):
        shutil.move(str(f), str(dd / f.name))
    inner.rmdir()
    tmp.rename(dd / 'replaced.onnx')
    return f'{part_dir.name}: flattened'

if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(flatten(Path(p)), flush=True)
