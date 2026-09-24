#!/usr/bin/env python3
"""bitmoodmouth — Python mouth extracted from jdnitrap/bitmood.

Context-mixing bit predictor used to train a memory and generate bytes.
Not a line-for-line port of the C++ engine (no LSTM / graph / SNN / image / audio).
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


PROB_ONE = 65536
PROB_MIN = 1
PROB_MAX = 65535


def clampi(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def stretch(p: int) -> float:
    q = clampi(p, PROB_MIN, PROB_MAX) / PROB_ONE
    q = min(max(q, 1e-6), 1.0 - 1e-6)
    return math.log(q / (1.0 - q))


def squash(x: float) -> float:
    x = min(max(x, -30.0), 30.0)
    return 1.0 / (1.0 + math.exp(-x))


def to_p16(p: float) -> int:
    return clampi(int(round(p * PROB_ONE)), PROB_MIN, PROB_MAX)


def bit_cost(p: int, bit: int) -> float:
    q = clampi(p, PROB_MIN, PROB_MAX) / PROB_ONE
    return -math.log2(q if bit else (1.0 - q))


def is_alpha(b: int) -> bool:
    return (65 <= b <= 90) or (97 <= b <= 122)


def is_digit(b: int) -> bool:
    return 48 <= b <= 57


def is_space(b: int) -> bool:
    return b in (9, 10, 13, 32)


def is_punct(b: int) -> bool:
    return (33 <= b <= 47) or (58 <= b <= 64) or (91 <= b <= 96) or (123 <= b <= 126)


class Adaptive:
    """n0/n1 counter → P(bit=1)."""

    def __init__(self) -> None:
        self.n0 = 1
        self.n1 = 1

    def p16(self) -> int:
        return to_p16(self.n1 / (self.n0 + self.n1))

    def update(self, bit: int) -> None:
        if bit:
            self.n1 += 1
        else:
            self.n0 += 1
        if self.n0 + self.n1 > 256:
            self.n0 = max(1, self.n0 // 2)
            self.n1 = max(1, self.n1 // 2)


class Mixer:
    def __init__(self, n: int) -> None:
        self.w = [0.0] * n

    def mix(self, stretched: List[float]) -> int:
        s = 0.0
        for wi, xi in zip(self.w, stretched):
            s += wi * xi
        return to_p16(squash(s))

    def update(self, stretched: List[float], bit: int, p: int, rate: float = 0.02) -> None:
        err = (1.0 if bit else 0.0) - (p / PROB_ONE)
        for i, xi in enumerate(stretched):
            self.w[i] += rate * err * xi


class Model:
    """Order-0..4 + class specialists, mixed per bit."""

    ORDERS = (0, 1, 2, 3, 4)

    def __init__(self) -> None:
        self.tables: Dict[int, Dict[Tuple, Adaptive]] = {o: defaultdict(Adaptive) for o in self.ORDERS}
        self.class_table: Dict[Tuple[int, int], Adaptive] = defaultdict(Adaptive)
        self.mixer = Mixer(len(self.ORDERS) + 1)
        self.history: bytearray = bytearray()
        self.bytes_learned = 0
        self.bits_spent = 0.0
        self.seen = [False] * 256

    def _ctx(self, order: int, bit_index: int, partial: int) -> Tuple:
        if order == 0:
            return (bit_index, partial)
        tail = bytes(self.history[-order:])
        return (bit_index, partial, tail)

    def _class_ctx(self, bit_index: int, partial: int) -> Tuple[int, int]:
        if not self.history:
            flags = 0
        else:
            b = self.history[-1]
            flags = (
                (1 if is_alpha(b) else 0)
                | (2 if is_digit(b) else 0)
                | (4 if is_space(b) else 0)
                | (8 if is_punct(b) else 0)
                | (16 if 65 <= b <= 90 else 0)
            )
        return (flags, (bit_index << 8) | partial)

    def predict_bit(self, bit_index: int, partial: int) -> Tuple[int, List[float], List[Adaptive]]:
        cells: List[Adaptive] = []
        stretched: List[float] = []
        for order in self.ORDERS:
            cell = self.tables[order][self._ctx(order, bit_index, partial)]
            cells.append(cell)
            stretched.append(stretch(cell.p16()))
        ccell = self.class_table[self._class_ctx(bit_index, partial)]
        cells.append(ccell)
        stretched.append(stretch(ccell.p16()))
        p = self.mixer.mix(stretched)
        return p, stretched, cells

    def learn_bit(self, bit: int, bit_index: int, partial: int) -> float:
        p, stretched, cells = self.predict_bit(bit_index, partial)
        for cell in cells:
            cell.update(bit)
        self.mixer.update(stretched, bit, p)
        cost = bit_cost(p, bit)
        self.bits_spent += cost
        return cost

    def learn_byte(self, b: int) -> float:
        cost = 0.0
        partial = 0
        for i in range(8):
            bit = (b >> (7 - i)) & 1
            cost += self.learn_bit(bit, i, partial)
            partial = (partial << 1) | bit
        self.history.append(b)
        self.seen[b] = True
        self.bytes_learned += 1
        return cost

    def feed_byte(self, b: int) -> None:
        """Advance context without learning (prompt / generation)."""
        self.history.append(b)

    def byte_distribution(self) -> List[float]:
        """P(byte) from the 8-bit tree of current context (no updates)."""
        mass = [0.0] * 256

        def walk(bit_index: int, partial: int, prob: float) -> None:
            if bit_index == 8:
                mass[partial] = prob
                return
            p1 = self.predict_bit(bit_index, partial)[0] / PROB_ONE
            walk(bit_index + 1, (partial << 1) | 1, prob * p1)
            walk(bit_index + 1, (partial << 1) | 0, prob * (1.0 - p1))

        walk(0, 0, 1.0)
        return mass


def sample_byte(mass: List[float], rng: random.Random, temp: float, top_k: int, charset: str, seen: List[bool]) -> int:
    weights = []
    for b, p in enumerate(mass):
        if charset == "ascii" and b > 127:
            continue
        if charset == "seen" and not seen[b]:
            continue
        if charset == "utf8":
            if b == 0:
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


def save_state(path: Path, model: Model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(model, f, protocol=4)
    tmp.replace(path)


def load_state(path: Path) -> Model:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Model):
        raise SystemExit(f"{path} is not a bitmoodmouth memory")
    return obj


def cmd_train(args: argparse.Namespace) -> int:
    path = Path(args.state)
    if path.exists():
        model = load_state(path)
        created = False
    else:
        model = Model()
        created = True
    print(
        f"{'new' if created else 'loaded'} memory {path}"
        + ("" if created else f" ({model.bytes_learned} bytes learned so far)"),
        file=sys.stderr,
    )
    for f in args.files:
        data = Path(f).read_bytes()
        bits = 0.0
        for b in data:
            bits += model.learn_byte(b)
        bpb = bits / len(data) if data else 0.0
        print(f"  {f:<40} {len(data):10d} bytes  {bpb:6.3f} bits/byte", file=sys.stderr)
    save_state(path, model)
    avg = model.bits_spent / model.bytes_learned if model.bytes_learned else 0.0
    print(
        f"saved {path}: {model.bytes_learned} bytes learned in total, {avg:.3f} bits/byte average",
        file=sys.stderr,
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    model = load_state(Path(args.memory))
    avg = model.bits_spent / model.bytes_learned if model.bytes_learned else 0.0
    print("type           text")
    print(f"bytes learned  {model.bytes_learned}")
    print(f"bits/byte      {avg:.3f} average while learning")
    print(f"history        {len(model.history)} bytes kept")
    print("specialists    O0 O1 O2 O3 O4 C")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
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
    n = args.nbytes
    while len(out) < n:
        mass = model.byte_distribution()
        b = sample_byte(mass, rng, args.temp, args.top_k, charset, model.seen)
        out.append(b)
        model.feed_byte(b)

    text = (prompt if args.state else b"") + out
    sys.stdout.buffer.write(text)
    if not text.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")
    return 0


def cmd_write(args: argparse.Namespace) -> int:
    path = Path(args.state)
    model = load_state(path) if path.exists() else Model()
    print(f"{'new' if not path.exists() else 'memory'} {path}", file=sys.stderr)
    print("type a line, Enter learns it; empty line + Enter quits. Ctrl-D also quits.", file=sys.stderr)
    learned = 0
    try:
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            if line.strip() == "" and learned:
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
    else:
        print(f"learned {learned} bytes (not saved)", file=sys.stderr)
    if args.out:
        Path(args.out).write_bytes(bytes(model.history))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bitmoodmouth", description="bitmood generator (Python mouth)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="learn files into a memory")
    t.add_argument("--state", required=True)
    t.add_argument("files", nargs="+")

    g = sub.add_parser("generate", help="sample new bytes")
    g.add_argument("nbytes", type=int)
    g.add_argument("prompt", nargs="?", default=None)
    g.add_argument("--state")
    g.add_argument("--temp", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--seed", type=int, default=0xC0FFEE)
    g.add_argument("--charset", choices=("seen", "utf8", "ascii", "any"), default="seen")

    i = sub.add_parser("info", help="describe a memory")
    i.add_argument("memory")

    w = sub.add_parser("write", help="type lines; each finished line is learned")
    w.add_argument("--state", required=True)
    w.add_argument("--out")
    w.add_argument("--no-save", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "info":
        return cmd_info(args)
    if args.cmd == "write":
        return cmd_write(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
