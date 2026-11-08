"""検証専用: 研究フォルダを含むパスを開いたら記録して失敗にする audit hook。
検証のときだけ PYTHONPATH にこのフォルダを入れる。通常の実行では使わない。"""
import os, sys

_FORBIDDEN = os.environ.get('MOSSNP_FORBIDDEN_SUBSTRING', '')
_LOG = os.environ.get('MOSSNP_AUDIT_LOG', '')

def _hook(event, args):
    if not _FORBIDDEN:
        return
    if event in ('open', 'import'):
        try:
            target = str(args[0]) if args else ''
        except Exception:
            target = ''
        if _FORBIDDEN in target:
            msg = f'FORBIDDEN {event} {target}'
            if _LOG:
                with open(_LOG, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')
            raise RuntimeError(msg)

if _FORBIDDEN:
    sys.addaudithook(_hook)
