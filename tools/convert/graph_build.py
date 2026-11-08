"""ONNX グラフ組み立ての共通部。重みは fp32 で混ぜた merged だけを使う。"""
from __future__ import annotations
import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper

DIM, FFN, HEADS, HEAD_DIM = 3072, 8192, 24, 128
EPS = 1e-6

class GraphBuilder:
    def __init__(self):
        self.nodes, self.inits, self.count = [], [], 0
    def name(self, base):
        self.count += 1
        return f'{base}_{self.count}'
    def const(self, base, array):
        name = self.name(base)
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name
    def op(self, op_type, inputs, base, domain=None, **attrs):
        out = self.name(base)
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=f'{base}/{op_type}_{self.count}', domain=domain, **attrs))
        return out
    def rmsnorm(self, x, weight, base):
        return self.op('SimplifiedLayerNormalization', [x, self.const(f'{base}.weight', weight.detach().numpy())],
                       f'{base}/rmsnorm', axis=-1, epsilon=1e-6)

def std_rmsnorm(g, x, gamma, base):
    sq = g.op('Mul', [x, x], f'{base}/sq')
    mean = g.op('ReduceMean', [sq], f'{base}/mean', axes=[-1], keepdims=1)
    rs = g.op('Sqrt', [g.op('Add', [mean, g.const(f'{base}.eps', np.array(EPS, np.float32))], f'{base}/eps')], f'{base}/sqrt')
    nx = g.op('Div', [x, rs], f'{base}/div')
    return g.op('Mul', [nx, g.const(f'{base}.gamma', gamma.detach().numpy())], f'{base}/gamma')
