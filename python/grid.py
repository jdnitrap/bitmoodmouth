"""ImHex-style grid specialist (E), copied as an idea from jdnitrap/bitmood.

Views:
  0  line rows   -- a row ends at each newline
  1  auto width 0 -- best repeating distance in 2..1024
  2  auto width 1 -- second best distance, not a multiple of the first

Each view votes twice:
  (byte above, byte above-right)
  (byte above, byte to the left, column)
"""

from __future__ import annotations

from typing import List, Tuple

MAX_WIDTH = 1024
PICK_EVERY = 128
WIDTH_DECAY = 1.0 - 1.0 / 2048.0
N_VIEWS = 3


class WidthFinder:
    """Hex-editor column slider: score distances 2..1024, keep two widths."""

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
    def __init__(self) -> None:
        self.widths = WidthFinder()
        self.line_start = 0
        self.prev_line_start = 0
        self.view_hits = [0, 0, 0]

    def labels(self) -> List[str]:
        labels = ["line rows"]
        for i in range(2):
            w = self.widths.width[i]
            labels.append(f"width {w}" if w else "no width yet")
        return labels

    def _byte_at(self, history: bytearray, abs_pos: int, limit: int) -> int:
        if abs_pos < 0 or abs_pos >= limit or abs_pos >= len(history):
            return 256
        return history[abs_pos]

    def cell(self, history: bytearray, view: int) -> GridCell:
        c = GridCell()
        pos = len(history)
        if view == 0:
            c.valid = True
            c.col = min(pos - self.line_start, 255)
            q = self.prev_line_start + (pos - self.line_start)
            c.above = self._byte_at(history, q, self.line_start)
            c.above_right = self._byte_at(history, q + 1, self.line_start)
            return c
        w = self.widths.width[view - 1]
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
        for i in range(N_VIEWS):
            cell = self.cell(history, i)
            vid = i if cell.valid else i + 0x80
            keys.append((1, vid, cell.above, cell.above_right, bit_index, partial))
            keys.append((2, vid, cell.above, last, cell.col, bit_index, partial))
        return keys

    def after_byte(self, history: bytearray, b: int) -> None:
        for i in range(N_VIEWS):
            cell = self.cell(history, i)
            if cell.valid and cell.above == b:
                self.view_hits[i] += 1
        if b == 10:
            self.prev_line_start = self.line_start
            self.line_start = len(history)
        self.widths.update(history)
