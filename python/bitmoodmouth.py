#!/usr/bin/env python3
"""bitmoodmouth — Python mouth from jdnitrap/bitmood.

Orders + class + grid E, with a mixer weight set per winning grid view.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import GridState, N_VIEWS

PROB_ONE = 65536
PROB_MIN = 1
PROB_MAX = 65535
N_GRID = N_VIEWS * 2
N_BASE = 5 + 1
N_IN = N_BASE + N_GRID


def clampi(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def stretch(p):
    q = clampi(p, PROB_MIN, PROB_MAX) / PROB_ONE
    q = min(max(q, 1e-6), 1.0 - 1e-6)
    return math.log(q / (1.0 - q))


def squash(x):
    x = min(max(x, -30.0), 30.0)
    return 1.0 / (1.0 + math.exp(-x))


def to_p16(p):
    return clampi(int(round(p * PROB_ONE)), PROB_MIN, PROB_MAX)


def bit_cost(p, bit):
    q = clampi(p, PROB_MIN, PROB_MAX) / PROB_ONE
    return -math.log2(q if bit else (1.0 - q))


def flags_of(b):
    return (
        (1 if (65 <= b <= 90) or (97 <= b <= 122) else 0)
        | (2 if 48 <= b <= 57 else 0)
        | (4 if b in (9, 10, 13, 32) else 0)
        | (8 if (33 <= b <= 47) or (58 <= b <= 64) or (91 <= b <= 96) or (123 <= b <= 126) else 0)
        | (16 if 65 <= b <= 90 else 0)
    )


class Adaptive:
    def __init__(self):
        self.n0 = 1
        self.n1 = 1

    def p16(self):
        return to_p16(self.n1 / (self.n0 + self.n1))

    def update(self, bit):
        if bit:
            self.n1 += 1
        else:
            self.n0 += 1
        if self.n0 + self.n1 > 256:
            self.n0 = max(1, self.n0 // 2)
            self.n1 = max(1, self.n1 // 2)


class Mixer:
    def __init__(self, n):
        self.w = [0.0] * n

    def ensure(self, n):
        if len(self.w) < n:
            self.w.extend([0.0] * (n - len(self.w)))

    def mix(self, stretched):
        self.ensure(len(stretched))
        s = sum(wi * xi for wi, xi in zip(self.w, stretched))
        return to_p16(squash(s))

    def update(self, stretched, bit, p, rate=0.02):
        self.ensure(len(stretched))
        err = (1.0 if bit else 0.0) - (p / PROB_ONE)
        for i, xi in enumerate(stretched):
            self.w[i] += rate * err * xi


class Model:
    ORDERS = (0, 1, 2, 3, 4)

    def __init__(self, fixed_width=0):
        self.tables = {o: defaultdict(Adaptive) for o in self.ORDERS}
        self.class_table = defaultdict(Adaptive)
        self.grid_table = defaultdict(Adaptive)
        self.mixers = [Mixer(N_IN) for _ in range(N_VIEWS)]
        self.history = bytearray()
        self.bytes_learned = 0
        self.bits_spent = 0.0
        self.seen = [False] * 256
        self.grid = GridState(fixed_width=fixed_width)
        self.use_grid = True

    def _ensure(self):
        if not hasattr(self, "grid"):
            self.grid = GridState()
        if not hasattr(self, "grid_table"):
            self.grid_table = defaultdict(Adaptive)
        if not hasattr(self, "use_grid"):
            self.use_grid = True
        if not hasattr(self, "mixers"):
            self.mixers = [Mixer(N_IN)]
        while len(self.mixers) < N_VIEWS:
            self.mixers.append(Mixer(N_IN))
        for mx in self.mixers:
            mx.ensure(N_IN)

    def _winner(self):
        return self.grid.winner if self.use_grid else 0

    def _ctx(self, order, bit_index, partial):
        if order == 0:
            return (bit_index, partial)
        return (bit_index, partial, bytes(self.history[-order:]))

    def _class_ctx(self, bit_index, partial):
        flags = flags_of(self.history[-1]) if self.history else 0
        return (flags, (bit_index << 8) | partial)

    def predict_bit(self, bit_index, partial):
        self._ensure()
        cells = []
        stretched = []
        for order in self.ORDERS:
            cell = self.tables[order][self._ctx(order, bit_index, partial)]
            cells.append(cell)
            stretched.append(stretch(cell.p16()))
        ccell = self.class_table[self._class_ctx(bit_index, partial)]
        cells.append(ccell)
        stretched.append(stretch(ccell.p16()))
        if self.use_grid:
            for key in self.grid.keys(self.history, bit_index, partial):
                gcell = self.grid_table[key]
                cells.append(gcell)
                stretched.append(stretch(gcell.p16()))
        mx = self.mixers[self._winner()]
        return mx.mix(stretched), stretched, cells, mx

    def learn_bit(self, bit, bit_index, partial):
        p, stretched, cells, mx = self.predict_bit(bit_index, partial)
        for cell in cells:
            cell.update(bit)
        mx.update(stretched, bit, p)
        cost = bit_cost(p, bit)
        self.bits_spent += cost
        return cost

    def learn_byte(self, b):
        self._ensure()
        cost = 0.0
        partial = 0
        for i in range(8):
            bit = (b >> (7 - i)) & 1
            cost += self.learn_bit(bit, i, partial)
            partial = (partial << 1) | bit
        self.history.append(b)
        self.seen[b] = True
        self.bytes_learned += 1
        self.grid.after_byte(self.history, b)
        return cost

    def feed_byte(self, b):
        self._ensure()
        self.history.append(b)
        self.grid.after_byte(self.history, b)

    def byte_distribution(self):
        mass = [0.0] * 256

        def walk(bit_index, partial, prob):
            if bit_index == 8:
                mass[partial] = prob
                return
            p1 = self.predict_bit(bit_index, partial)[0] / PROB_ONE
            walk(bit_index + 1, (partial << 1) | 1, prob * p1)
            walk(bit_index + 1, (partial << 1) | 0, prob * (1.0 - p1))

        walk(0, 0, 1.0)
        return mass


def sample_byte(mass, rng, temp, top_k, charset, seen):
    weights = []
    for b, p in enumerate(mass):
        if charset == "ascii" and b > 127:
            continue
        if charset == "seen" and not seen[b]:
            continue
        if charset == "utf8" and b == 0:
            continue
        if p <= 0:
            continue
        logit = math.log(max(p, 1e-12)) / max(temp, 1e-6)
        weights.append((b, math.exp(logit)))
    if not weights:
        weights = [(b, max(mass[b], 1e-12)) for b in range(32, 127)]
    weights.sort(key=lambda kv: kv[1], reverse=True)
    if top_k > 0:
        weights = weights[:top_k]
    total = sum(w for _, w in weights)
    if total <= 0:
        return 32
    r = rng.random() * total
    acc = 0.0
    for b, w in weights:
        acc += w
        if r <= acc:
            return b
    return weights[-1][0]


def save_state(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(model, f, protocol=4)
    tmp.replace(path)


def load_state(path):
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Model):
        raise SystemExit(f"{path} is not a bitmoodmouth memory")
    obj._ensure()
    return obj


def cmd_train(args):
    path = Path(args.state)
    existed = path.exists()
    model = load_state(path) if existed else Model(fixed_width=args.row)
    model.use_grid = not args.no_grid
    if args.row:
        model.grid.fixed_width = args.row
    print(
        f"{'loaded' if existed else 'new'} memory {path}"
        + (f" ({model.bytes_learned} bytes learned so far)" if existed else ""),
        file=sys.stderr,
    )
    for f in args.files:
        data = Path(f).read_bytes()
        bits = sum(model.learn_byte(b) for b in data)
        bpb = bits / len(data) if data else 0.0
        print(f"  {f:<40} {len(data):10d} bytes  {bpb:6.3f} bits/byte", file=sys.stderr)
    print("grid views     " + "  ".join(f"[{lab}]" for lab in model.grid.labels()), file=sys.stderr)
    print("grid winner    " + "  ".join(str(n) for n in model.grid.winner_counts), file=sys.stderr)
    save_state(path, model)
    avg = model.bits_spent / model.bytes_learned if model.bytes_learned else 0.0
    print(f"saved {path}: {model.bytes_learned} bytes, {avg:.3f} bits/byte average", file=sys.stderr)
    return 0


def cmd_info(args):
    model = load_state(Path(args.memory))
    avg = model.bits_spent / model.bytes_learned if model.bytes_learned else 0.0
    print("type           text")
    print(f"bytes learned  {model.bytes_learned}")
    print(f"bits/byte      {avg:.3f} average while learning")
    print(f"history        {len(model.history)} bytes kept")
    print("specialists    O0-O4 C E(line,auto,word,fixed; mixer per winning view)")
    print("grid views     " + "  ".join(f"[{lab}]" for lab in model.grid.labels()))
    print("grid hits      " + "  ".join(str(n) for n in model.grid.view_hits))
    print("grid winner    " + "  ".join(str(n) for n in model.grid.winner_counts))
    print(f"current view   {model.grid.winner}  {model.grid.labels()[model.grid.winner]}")
    return 0


def cmd_generate(args):
    rng = random.Random(args.seed)
    if args.state:
        model = load_state(Path(args.state))
        prompt = args.prompt.encode("utf-8") if args.prompt is not None else b""
        for b in prompt:
            model.feed_byte(b)
    else:
        model = Model()
        prompt = (args.prompt or "The quick brown fox ").encode("utf-8")
        for b in prompt:
            model.learn_byte(b)
    charset = args.charset
    if charset == "seen" and not any(model.seen):
        charset = "utf8"
    out = bytearray()
    while len(out) < args.nbytes:
        b = sample_byte(model.byte_distribution(), rng, args.temp, args.top_k, charset, model.seen)
        out.append(b)
        model.feed_byte(b)
    text = (prompt if args.state else b"") + out
    sys.stdout.buffer.write(text)
    if not text.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")
    return 0


def cmd_write(args):
    path = Path(args.state)
    model = load_state(path) if path.exists() else Model()
    print(f"{'new' if not path.exists() else 'memory'} {path}", file=sys.stderr)
    print("type a line, Enter learns it; empty line + Enter quits.", file=sys.stderr)
    learned = 0
    try:
        while True:
            line = sys.stdin.readline()
            if line == "" or (line.strip() == "" and learned):
                break
            data = line.encode("utf-8")
            for b in data:
                model.learn_byte(b)
            learned += len(data)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    if not args.no_save:
        save_state(path, model)
        print(f"learned {learned} bytes; saved {path}", file=sys.stderr)
    if args.out:
        Path(args.out).write_bytes(bytes(model.history))
    return 0


def cmd_compare(args):
    data = Path(args.file).read_bytes()
    on = Model(fixed_width=args.row)
    off = Model(fixed_width=args.row)
    off.use_grid = False
    bits_on = sum(on.learn_byte(b) for b in data)
    bits_off = sum(off.learn_byte(b) for b in data)
    n = len(data) or 1
    print(f"file           {args.file}")
    print(f"bytes          {len(data)}")
    print(f"with grid      {bits_on / n:.3f} bits/byte")
    print(f"without grid   {bits_off / n:.3f} bits/byte")
    print(f"delta          {(bits_off - bits_on) / n:+.3f} bits/byte (positive = grid helped)")
    print("views          " + "  ".join(f"[{lab}]" for lab in on.grid.labels()))
    print("hits           " + "  ".join(str(x) for x in on.grid.view_hits))
    print("winner ticks   " + "  ".join(str(x) for x in on.grid.winner_counts))
    if args.other:
        other = Path(args.other).read_bytes()
        bits = sum(on.learn_byte(b) for b in other)
        print(f"then {args.other}  {len(other)} bytes  {bits / (len(other) or 1):.3f} bits/byte")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="bitmoodmouth")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--state", required=True)
    t.add_argument("--no-grid", action="store_true")
    t.add_argument("--row", type=int, default=0)
    t.add_argument("files", nargs="+")
    g = sub.add_parser("generate")
    g.add_argument("nbytes", type=int)
    g.add_argument("prompt", nargs="?", default=None)
    g.add_argument("--state")
    g.add_argument("--temp", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--seed", type=int, default=0xC0FFEE)
    g.add_argument("--charset", choices=("seen", "utf8", "ascii", "any"), default="seen")
    i = sub.add_parser("info")
    i.add_argument("memory")
    w = sub.add_parser("write")
    w.add_argument("--state", required=True)
    w.add_argument("--out")
    w.add_argument("--no-save", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("file")
    c.add_argument("--other")
    c.add_argument("--row", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {
        "train": cmd_train,
        "generate": cmd_generate,
        "info": cmd_info,
        "write": cmd_write,
        "compare": cmd_compare,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
