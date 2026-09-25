"""ImHex-style grid specialist (E) from jdnitrap/bitmood.

Views:
  line rows, two auto widths, word 2/4/8, optional fixed row width.

Each view votes twice: (above, above-right) and (above, left, column).
A tracker scores which view's above matches the real byte and names a winner.
"""

from __future__ import annotations

from typing import List, Tuple

MAX_WIDTH = 1024
PICK_EVERY = 128
WIDTH_DECAY = 1.0 - 1.0 / 2048.0
VIEW_DECAY = 1.0 - 1.0 / 64.0

KIND_LINE = "line"
KIND_AUTO = "auto"
KIND_WORD = "word"
KIND_FIXED = "fixed"


class WidthFinder:
    def __init__(self) -> None:
        self.score = [0.0] * (MAX_WIDTH + 1)
        self.width = [0, 0]
        self.since_pick = 0

    def update(self, history: bytearray) -> None:
        n = len(history)
        if n < 3:
            return
        now = history[-1]
        top = min(MAX_WIDTH, n - 1)
        for d in range(2, top + 1):
            hit = 1.0 if history[-(d + 1)] == now else 0.0
            self.score[d] = self.score[d] * WIDTH_DECAY + hit
        self.since_pick += 1
        if self.since_pick >= PICK_EVERY:
            self.since_pick = 0
            self.pick()

    def pick(self) -> None:
        mean = sum(self.score[2:]) / (MAX_WIDTH - 1)

        def stands_out(d: int) -> bool:
            return self.score[d] >= 4.0 and self.score[d] >= 1.25 * mean

        def related(a: int, b: int) -> bool:
            return a % b == 0 or b % a == 0

        def smallest_near_top(skip_related_to: int) -> int:
            top = 0.0
            for d in range(2, MAX_WIDTH + 1):
                if skip_related_to and related(d, skip_related_to):
                    continue
                if self.score[d] > top:
                    top = self.score[d]
            if top <= 0.0:
                return 0
            for d in range(2, MAX_WIDTH + 1):
                if skip_related_to and related(d, skip_related_to):
                    continue
                if self.score[d] >= 0.97 * top:
                    return d
            return 0

        best = smallest_near_top(0)
        second = smallest_near_top(best) if best else 0
        self.width[0] = best if best and stands_out(best) else 0
        self.width[1] = second if second and stands_out(second) else 0


class GridCell:
    def __init__(self) -> None:
        self.valid = False
        self.above = 256
        self.above_right = 256
        self.col = 0
        self.width = 0


class View:
    def __init__(self, kind: str, param: int = 0) -> None:
        self.kind = kind
        self.param = param


def default_views(data_type: str = "text", fixed_width: int = 0) -> List[View]:
    views = [View(KIND_LINE), View(KIND_AUTO, 0), View(KIND_AUTO, 1)]
    if data_type in ("raw", "text"):
        views.extend([View(KIND_WORD, 2), View(KIND_WORD, 4), View(KIND_WORD, 8)])
    if fixed_width > 0:
        views.append(View(KIND_FIXED, fixed_width))
    return views


class GridState:
    def __init__(self, data_type: str = "text", fixed_width: int = 0) -> None:
        self.views = default_views(data_type, fixed_width)
        self.widths = WidthFinder()
        self.line_start = 0
        self.prev_line_start = 0
        self.view_hits = [0] * len(self.views)
        self.view_score = [0.0] * len(self.views)
        self.winner = -1

    def n_views(self) -> int:
        return len(self.views)

    def n_votes(self) -> int:
        return 2 * len(self.views)

    def label(self, i: int) -> str:
        v = self.views[i]
        if v.kind == KIND_LINE:
            return "line rows"
        if v.kind == KIND_AUTO:
            w = self.widths.width[v.param]
            return f"width {w}" if w else "no width yet"
        if v.kind == KIND_WORD:
            return f"word {v.param}"
        if v.kind == KIND_FIXED:
            return f"row {v.param}"
        return "?"

    def labels(self) -> List[str]:
        return [self.label(i) for i in range(len(self.views))]

    def best_label(self) -> str:
        return "no grid" if self.winner < 0 else self.label(self.winner)

    def _byte_at(self, history: bytearray, abs_pos: int, limit: int) -> int:
        if abs_pos < 0 or abs_pos >= limit or abs_pos >= len(history):
            return 256
        return history[abs_pos]

    def _fixed_width(self, v: View) -> int:
        if v.kind == KIND_AUTO:
            return self.widths.width[v.param]
        if v.kind in (KIND_WORD, KIND_FIXED):
            return v.param
        return 0

    def cell(self, history: bytearray, view: int) -> GridCell:
        c = GridCell()
        pos = len(history)
        v = self.views[view]
        if v.kind == KIND_LINE:
            c.valid = True
            c.col = min(pos - self.line_start, 255)
            q = self.prev_line_start + (pos - self.line_start)
            c.above = self._byte_at(history, q, self.line_start)
            c.above_right = self._byte_at(history, q + 1, self.line_start)
            return c
        w = self._fixed_width(v)
        if w <= 0 or pos < w:
            return c
        c.valid = True
        c.width = w
        c.col = min(pos % w, 1023)
        c.above = self._byte_at(history, pos - w, pos)
        c.above_right = self._byte_at(history, pos - w + 1, pos)
        return c

    def keys(self, history: bytearray, bit_index: int, partial: int) -> List[Tuple]:
        last = history[-1] if history else 256
        keys: List[Tuple] = []
        for i in range(len(self.views)):
            cell = self.cell(history, i)
            vid = i if cell.valid else i + 0x80
            keys.append((1, vid, cell.above, cell.above_right, bit_index, partial))
            keys.append((2, vid, cell.above, last, cell.col, bit_index, partial))
        return keys

    def after_byte(self, history: bytearray, b: int) -> None:
        best_i = -1
        best_s = 0.0
        for i in range(len(self.views)):
            cell = self.cell(history, i)
            hit = 1.0 if cell.valid and cell.above == b else 0.0
            if hit:
                self.view_hits[i] += 1
            self.view_score[i] = self.view_score[i] * VIEW_DECAY + hit
            if cell.valid and self.view_score[i] > best_s:
                best_s = self.view_score[i]
                best_i = i
        self.winner = best_i if best_s >= 0.15 else -1
        if b == 10:
            self.prev_line_start = self.line_start
            self.line_start = len(history)
        self.widths.update(history)
