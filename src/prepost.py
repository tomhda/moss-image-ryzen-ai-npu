"""DiT の前処理・後処理（fp32）と DD セッションの共通処理。

前/後処理の式は公式 pipeline と同じ。重みは利用者の MOSS フォルダの transformer から読む。
"""
from __future__ import annotations
import ctypes, gc, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import npu_config as cfg

EP_NAME = 'RyzenAILightExecutionProvider'
DIM, FFN, HEADS, HEAD_DIM = 3072, 8192, 24, 128
BLOCK_MODS = ('shift_a', 'scale_a', 'gate_a', 'shift_f', 'scale_f', 'gate_f')
_dd_registered = False

def custom_op() -> Path:
    return cfg.ryzen_dll()

def register_dd():
    global _dd_registered
    if _dd_registered:
        return
    import onnxruntime as ort
    dll = custom_op()
    os.add_dll_directory(str(dll.parent))
    ctypes.CDLL(str(dll))
    prev = os.getcwd()
    try:
        os.chdir(dll.parent)
        ort.register_execution_provider_library(EP_NAME, str(dll))
    finally:
        os.chdir(prev)
    _dd_registered = True

def dd_session_dir(dd_dir: Path, onnx_name: str, intra=2, no_spin=False, no_arena=False):
    """dd_dir 直下の onnx を開く。compile_fusion_rt は 0 のまま開く。"""
    import onnxruntime as ort
    register_dd()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if intra:
        so.intra_op_num_threads = intra
    if no_spin:
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.add_session_config_entry('session.inter_op.allow_spinning', '0')
    if no_arena:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    so.add_provider_for_devices([d for d in ort.get_ep_devices() if d.ep_name == EP_NAME], {
        'dd_cache': (dd_dir / 'cache').as_posix(), 'onnx_custom_ops_const_key': '', 'compile_fusion_rt': '0'})
    so.register_custom_ops_library(str(custom_op()))
    return ort.InferenceSession(str(dd_dir / onnx_name), sess_options=so)

# ---------- pipeline の式 (fp32) ----------
def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin
def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)
class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight
class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)
    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))
class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])
    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x
class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self._default_bands(torch.device('cpu')))
    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)
    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

_prepost = None
def load_prepost():
    global _prepost
    if _prepost is not None:
        return _prepost
    torch.set_grad_enabled(False)
    from mmap_weights import open_folder
    w = open_folder(cfg.moss() / 'transformer')
    def lin(name):
        m = nn.Linear(w[name + '.weight'].shape[1], w[name + '.weight'].shape[0],
                      bias=(name + '.bias' in w))
        m.weight.data = w[name + '.weight'].float()
        if m.bias is not None:
            m.bias.data = w[name + '.bias'].float()
        return m.eval()
    def rms(name, dim):
        m = _RMSNorm(dim)
        m.weight.data = w[name + '.weight'].float() if (name + '.weight' in w) else torch.ones(dim)
        return m.eval()
    gen = {}
    gen['video_embedder'] = lin('video_embedder')
    gen['time0'] = lin('time_embedding.0')
    gen['time2'] = lin('time_embedding.2')
    gen['shared_mod'] = lin('shared_block_modulation')
    gen['head_mod'] = lin('video_head_modulation')
    gen['final_norm'] = rms('final_norm_video', DIM)
    fl = nn.Linear(DIM, 128, bias=False)
    fl.weight.data = w['final_linear_video.weight'].float()
    gen['final_linear'] = fl.eval()
    tp = _TextProjection(4096, DIM, FFN, HEADS)
    tp.proj.weight.data = w['text_embedder.proj.weight'].float()
    tp.proj.bias.data = w['text_embedder.proj.bias'].float()
    for bi in range(2):
        blk = tp.blocks[bi]
        p = f'text_embedder.blocks.{bi}.'
        blk.norm1.weight.data = w[p+'norm1.weight'].float()
        blk.norm2.weight.data = w[p+'norm2.weight'].float()
        blk.q_norm.weight.data = w[p+'q_norm.weight'].float()
        blk.k_norm.weight.data = w[p+'k_norm.weight'].float()
        for nm in ('qkv','proj','ffn_gate','ffn_up','ffn_down'):
            getattr(blk, nm).weight.data = w[p+nm+'.weight'].float()
            getattr(blk, nm).bias.data = w[p+nm+'.bias'].float()
    gen['text_embedder'] = tp.eval()
    gen['rope_mod'] = _RoPE(HEAD_DIM, 10000.0)
    _prepost = gen
    return gen

def time_mod(gen, timestep):
    times = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1).float()
    half = 128
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = gen['time2'](F.silu(gen['time0'](embedding)))
    time = time.float()
    mod = gen['shared_mod'](time).view(-1, 6, DIM)
    head = gen['head_mod'](time).view(-1, 2, DIM)
    return mod, head

def preprocess_np(gen, latents_t, timestep, prompt_embeds_t, height=32, width=32):
    mod, head = time_mod(gen, timestep)
    h, w = height, width
    S = prompt_embeds_t.shape[1]
    video = latents_t.flatten(2).transpose(1, 2)
    video = gen['video_embedder'](video.contiguous())
    rope = gen['rope_mod'](h, w, S, latents_t.device).to(video.dtype)
    text = gen['text_embedder'](prompt_embeds_t, rope[:, h*w:])
    x = torch.cat((video, text), 1)
    return x, mod, head, rope

def postprocess_np(gen, block41_out_t, head_t, height=32, width=32):
    hw = height * width
    x = gen['final_norm'](block41_out_t[:, :hw])
    head = head_t
    if head.ndim == 2:
        head = head.view(-1, 2, head.shape[-1] // 2)
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = gen['final_linear'](x)
    return x.transpose(1, 2).reshape(1, -1, height, width)

def mem_snapshot():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        vm = psutil.virtual_memory()
        d = {'rss_mb': mi.rss/2**20, 'private_mb': (mi.private/2**20 if hasattr(mi,'private') else None),
             'avail_mb': vm.available/2**20, 'peak_wset_mb': (mi.peak_wset/2**20 if hasattr(mi,'peak_wset') else None)}
        return d
    except Exception as e:
        return {'error': str(e)}

def mods_of(modulation_np):
    return {nm: np.ascontiguousarray(modulation_np[:, i:i+1]).astype(np.float32) for i, nm in enumerate(BLOCK_MODS)}

_max_run = [0.0]
def run_timed(sess, out_names, feeds):
    t = time.perf_counter()
    r = sess.run(out_names, feeds)
    dt = (time.perf_counter() - t) * 1000
    _max_run[0] = max(_max_run[0], dt)
    return dict(zip(out_names, r)), dt

def release_sessions(sessions):
    try:
        del sessions
    except Exception:
        pass
    gc.collect()
