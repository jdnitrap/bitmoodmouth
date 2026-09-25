"""ImHex-style grid specialist (E), from jdnitrap/bitmood.

Views:
  0    line rows
  1,2  auto widths (two best distances in 2..1024)
  3,4,5 word 2 / 4 / 8
  6    optional fixed row width (image / record)

Each view votes twice:
  (byte above, byte above-right)
  (byte above, byte to the left, column)

A decayed hit score picks the winning view so the model can keep
a separate mixer weight set per winner.
"""

from __future__ import annotations

from typing import List, Tuple

MAX_WIDTH = 1024
PICK_EVERY = 128
WIDTH_DECAY = 1.0 - 1.0 / 2048.0
VIEW_DECAY = 1.0 - 1.0 / 64.0
WORD_WIDTHS = (2, 4, 8)
N_AUTO = 2
N_WORD = 3
N_VIEWS = 1 + N_AUTO + N_WORD + 1


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


class GridState:
    def __init__(self, fixed_width: int = 0) -> None:
        self.widths = WidthFinder()
        self.line_start = 0
        self.prev_line_start = 0
        self.view_hits = [0] * N_VIEWS
        self.view_score = [0.0] * N_VIEWS
        self.winner = 0
        self.winner_counts = [0] * N_VIEWS
        self.fixed_width = fixed_width

    def labels(self) -> List[str]:
        out = ["line rows"]
        for i in range(N_AUTO):
            w = self.widths.width[i]
            out.append(f"width {w}" if w else "no width yet")
        for w in WORD_WIDTHS:
            out.append(f"word {w}")
        out.append(f"row {self.fixed_width}" if self.fixed_width else "no fixed row")
        return out

    def _byte_at(self, history: bytearray, abs_pos: int, limit: int) -> int:
        if abs_pos < 0 or abs_pos >= limit or abs_pos >= len(history):
            return 256
        return history[abs_pos]

    def _fixed_cell(self, history: bytearray, w: int) -> GridCell:
        c = GridCell()
        pos = len(history)
        if w <= 0 or pos < w:
            return c
        c.valid = True
        c.width = w
        c.col = min(pos % w, 1023)
        c.above = self._byte_at(history, pos - w, pos)
        c.above_right = self._byte_at(history, pos - w + 1, pos)
        return c

    def cell(self, history: bytearray, view: int) -> GridCell:
        pos = len(history)
        if view == 0:
            c = GridCell()
            c.valid = True
            c.col = min(pos - self.line_start, 255)
            q = self.prev_line_start + (pos - self.line_start)
            c.above = self._byte_at(history, q, self.line_start)
            c.above_right = self._byte_at(history, q + 1, self.line_start)
            return c
        if view in (1, 2):
            return self._fixed_cell(history, self.widths.width[view - 1])
        if view in (3, 4, 5):
            return self._fixed_cell(history, WORD_WIDTHS[view - 3])
        return self._fixed_cell(history, self.fixed_width)

    def keys(self, history: bytearray, bit_index: int, partial: int) -> List[Tuple]:
        last = history[-1] if history else 256
        keys: List[Tuple] = []
        for i in range(N_VIEWS):
            cell = self.cell(history, i)
            vid = i if cell.valid else i + 0x80
            keys.append((1, vid, cell.above, cell.above_right, bit_index, partial))
            keys.append((2, vid, cell.above, last, cell.col, bit_index, partial))
        return keys

    def after_byte(self, history: bytearray, b: int) -> None:
        best_i = 0
        best_s = -1.0
        for i in range(N_VIEWS):
            cell = self.cell(history, i)
            hit = 1.0 if cell.valid and cell.above == b else 0.0
            if hit:
                self.view_hits[i] += 1
            self.view_score[i] = self.view_score[i] * VIEW_DECAY + hit
            if self.view_score[i] > best_s:
                best_s = self.view_score[i]
                best_i = i
        self.winner = best_i
        self.winner_counts[best_i] += 1
        if b == 10:
            self.prev_line_start = self.line_start
            self.line_start = len(history)
        self.widths.update(history)
