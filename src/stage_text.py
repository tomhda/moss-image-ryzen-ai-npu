"""文字段 (text env): トークン数をエンコーダの前に数え、上限を超える組は SKIP にする。
同じプロンプトは Out/cache を使い回す。キャッシュの鍵にはモデルの版・処理の版を入れる。"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import npu_config as cfg

PROC_VERSION = 'p7-text-1'

def cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system

def cache_name(key_material: str) -> str:
    return hashlib.sha256(key_material.encode('utf-8')).hexdigest()[:24] + '.safetensors'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--size', type=int, choices=[512, 1024], default=512)
    ap.add_argument('--run-id', required=True)
    a = ap.parse_args()
    max_tokens = 192 if a.size == 512 else 256
    t_start = time.perf_counter(); c_start = cpu_seconds()
    try:
        if a.threads:
            torch.set_num_threads(a.threads)
            torch.set_num_interop_threads(1)
        moss = cfg.moss()
        model_rev = os.environ.get('MOSSNP_MODEL_REV', 'unknown')
        out = Path(a.out); cache = out / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        items = json.loads(Path(a.items).read_text(encoding='utf-8'))
        # まず tokenizer だけで数える (エンコーダはまだ読み込まない)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(moss / 'tokenizer')
        template_id = hashlib.sha256(str(tokenizer.chat_template).encode('utf-8')).hexdigest()[:16]
        counted = {}  # prompt -> (tokens, formatted)
        for it in items:
            p = it['prompt']
            if p in counted:
                continue
            formatted = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': p}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True)
            ids = tokenizer(formatted, return_tensors='pt')['input_ids']
            counted[p] = (int(ids.shape[1]), formatted)
        index_path = cache / 'index.json'
        index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
        def key_for(prompt: str) -> str:
            return f'{model_rev}|{template_id}|{PROC_VERSION}|{prompt}'
        plan = []  # (item, tokens, cache_file or None)
        for it in items:
            tokens, _ = counted[it['prompt']]
            fn = index.get(key_for(it['prompt']))
            f = cache / fn if fn else None
            plan.append((it, tokens, f if (f and f.exists()) else None))
        ok_prompts = {it['prompt'] for it, tokens, _ in plan if 1 <= tokens <= max_tokens}
        misses = sorted({it['prompt'] for it, tokens, f in plan if f is None and it['prompt'] in ok_prompts})
        load_s = 0.0; per_prompt_s = {}
        if misses:
            from transformers import AutoConfig, AutoModel
            from mmap_weights import assign, open_folder
            from moss_model import materialize_rotary
            t0 = time.perf_counter()
            config = AutoConfig.from_pretrained(moss / 'text_encoder')
            with torch.device('meta'):
                model = AutoModel.from_config(config, dtype=torch.bfloat16)
            report = assign(model, open_folder(moss / 'text_encoder'))
            report['rotary_rebuilt'] = materialize_rotary(model)
            report['meta_left'] = [n for n, b in model.named_buffers() if b.is_meta]
            model.eval()
            if report['missing'] or any(not n.startswith('visual') for n in report['meta_left']):
                raise SystemExit('重みが足りない')
            load_s = time.perf_counter() - t0
            print(f'text model load {load_s:.2f}s misses {len(misses)}', flush=True)
            for prompt in misses:
                tokens, formatted = counted[prompt]
                inputs = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=4096)
                t1 = time.perf_counter()
                with torch.no_grad():
                    output = model(**inputs, use_cache=False, return_dict=True)
                hidden = output.last_hidden_state[0].contiguous()
                if not bool(torch.isfinite(hidden).all()):
                    raise SystemExit(f'文字エンコーダの出力が有限でない: {prompt[:40]!r}')
                per_prompt_s[prompt] = time.perf_counter() - t1
                fn = cache_name(key_for(prompt))
                tmp = cache / (fn + '.tmp')
                save_file({'input_ids': inputs['input_ids'][0].contiguous(),
                           'attention_mask': inputs['attention_mask'][0].contiguous(),
                           'last_hidden_state': hidden},
                          tmp, metadata={'prompt': prompt, 'formatted': formatted})
                os.replace(tmp, cache / fn)
                index[key_for(prompt)] = fn
                print(f'text encoded tokens {int(hidden.shape[0])} {per_prompt_s[prompt]:.2f}s', flush=True)
            tmp_index = index_path.with_suffix('.json.tmp')
            tmp_index.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
            os.replace(tmp_index, index_path)
        ok, failed, entries = [], [], []
        for it, tokens, _f in plan:
            if not (1 <= tokens <= max_tokens):
                print(f"SKIP id={it['id']} tokens={tokens}", flush=True)
                failed.append({'id': it['id'], 'seed': int(it['seed']), 'tokens': tokens, 'reason': f'tokens out of 1..{max_tokens}'})
                continue
            f = cache / index[key_for(it['prompt'])]
            with safe_open(str(f), framework='pt') as h:
                file_tokens = int(h.get_tensor('last_hidden_state').shape[0])
            if file_tokens != tokens:
                raise SystemExit(f'トークン数の不一致: {it["id"]} count={tokens} file={file_tokens}')
            e = {'id': it['id'], 'prompt': it['prompt'], 'seed': int(it['seed']),
                 'cache_file': str(f), 'tokens': tokens,
                 'cache_hit': it['prompt'] not in per_prompt_s,
                 'encode_s': round(per_prompt_s.get(it['prompt'], 0.0), 3)}
            ok.append(it); entries.append(e)
        manifest = {'run_id': a.run_id, 'size': a.size, 'items': entries, 'failed': failed}
        tmp_m = out / 'text_manifest.json.tmp'
        tmp_m.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp_m, out / 'text_manifest.json')
        wall = time.perf_counter() - t_start; cpu = cpu_seconds() - c_start
        log = {'run_id': a.run_id, 'size': a.size, 'threads': torch.get_num_threads(), 'load_s': round(load_s, 2),
               'wall_s': round(wall, 2), 'cpu_s': round(cpu, 2), 'items': entries, 'failed': failed}
        (out / 'text_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
        (out / f'text_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'text', 'run_id': a.run_id, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2)}), encoding='utf-8')
        print(f'text done ok {len(ok)} failed {len(failed)} load_s {load_s:.2f} wall {wall:.2f}s cpu {cpu:.2f}s', flush=True)
    except BaseException:
        try:
            wall = time.perf_counter() - t_start; cpu = cpu_seconds() - c_start
            (Path(a.out) / f'text_cpu_{a.run_id}.json').write_text(json.dumps({'stage': 'text', 'run_id': a.run_id, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2), 'failed': True}), encoding='utf-8')
        except Exception:
            pass
        raise

if __name__ == '__main__':
    main()
