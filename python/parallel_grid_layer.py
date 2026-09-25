#!/usr/bin/env python3
"""Tiny byte LM + parallel 2-D grid experts + router. See repo README."""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def line_above(seq: List[int], i: int) -> Tuple[int, int]:
    col = 0
    for j in range(i - 1, -1, -1):
        if seq[j] == 10:
            start = j + 1
            col = i - start
            prev_end = j
            prev_start = prev_end
            while prev_start > 0 and seq[prev_start - 1] != 10:
                prev_start -= 1
            q = prev_start + col
            if prev_start <= q < prev_end:
                return seq[q], col
            return 256, col
        col += 1
        if col > 255:
            break
    return 256, min(i, 255)


def width_above(seq: List[int], i: int, w: int) -> Tuple[int, int]:
    if w <= 0 or i < w:
        return 256, i % max(w, 1) if w else 0
    return seq[i - w], i % w


def best_period(seq: List[int], max_w: int = 64) -> int:
    n = len(seq)
    if n < 8:
        return 0
    best_w, best_s = 0, 0.0
    top = min(max_w, n // 2)
    for w in range(2, top + 1):
        hits = tot = 0
        for i in range(w, n):
            tot += 1
            if seq[i] == seq[i - w]:
                hits += 1
        s = hits / max(tot, 1)
        if s > best_s:
            best_s, best_w = s, w
    return best_w if best_s >= 0.15 else 0


class ExpertSpec:
    def __init__(self, name: str, kind: str, param: int = 0) -> None:
        self.name = name
        self.kind = kind
        self.param = param


DEFAULT_EXPERTS = [
    ExpertSpec("line", "line"),
    ExpertSpec("w8", "width", 8),
    ExpertSpec("w16", "width", 16),
    ExpertSpec("w19", "width", 19),
    ExpertSpec("w24", "width", 24),
    ExpertSpec("word4", "width", 4),
    ExpertSpec("auto", "auto"),
]


def expert_neighbors(seq: List[int], specs: List[ExpertSpec]):
    n = len(seq)
    e = len(specs)
    auto_w = best_period(seq)
    above = torch.zeros(n, e, dtype=torch.long)
    col = torch.zeros(n, e, dtype=torch.long)
    for i in range(n):
        for k, spec in enumerate(specs):
            if spec.kind == "line":
                a, c = line_above(seq, i)
            elif spec.kind == "auto":
                a, c = width_above(seq, i, auto_w)
            else:
                a, c = width_above(seq, i, spec.param)
            above[i, k] = a
            col[i, k] = min(c, 127)
    return above, col, auto_w


class TinyByteLM(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, n_experts=7, use_grid=True):
        super().__init__()
        self.use_grid = use_grid
        self.tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(512, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4, batch_first=True, dropout=0.0
        )
        self.core = nn.TransformerEncoder(enc, num_layers=layers)
        self.above_emb = nn.Embedding(257, d)
        self.col_emb = nn.Embedding(128, d)
        self.router = nn.Linear(d, n_experts)
        self.mix = nn.Linear(d, d)
        self.head = nn.Linear(d, 256)

    def forward(self, bytes_in, above, col):
        B, T = bytes_in.shape
        pos = torch.arange(T, device=bytes_in.device).clamp(max=511)
        h = self.tok(bytes_in) + self.pos(pos)[None, :, :]
        causal = torch.nn.Transformer.generate_square_subsequent_mask(T, device=bytes_in.device)
        h = self.core(h, mask=causal, is_causal=True)
        gates = None
        if self.use_grid:
            feat = self.above_emb(above) + self.col_emb(col.clamp(0, 127))
            gates = torch.softmax(self.router(h), dim=-1)
            mixed = (feat * gates.unsqueeze(-1)).sum(dim=2)
            h = h + self.mix(mixed)
        return self.head(h), gates
