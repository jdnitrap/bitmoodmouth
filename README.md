# bitmoodmouth

Generator (mouth) extracted from [jdnitrap/bitmood](https://github.com/jdnitrap/bitmood).

Python is the working tree. The C++ copy under `src/` is incomplete and not required to run.

## Python

No third-party packages. Python 3.8+.

```
python3 python/bitmoodmouth.py train --state brain.pkl notes.txt
python3 python/bitmoodmouth.py info brain.pkl
python3 python/bitmoodmouth.py generate 300 "The " --state brain.pkl --temp 0.8 --seed 42
python3 python/bitmoodmouth.py write --state brain.pkl --out draft.txt
```

`generate` never learns from its own output. `--state` memories are pickle files and are **not** compatible with C++ `brain.bin`.

Specialists: order 0-4, class vote C, and grid specialist E (ImHex idea): line rows plus two auto widths scored on distances 2..1024. Each view votes with the byte above and with (above, left, column). `train --no-grid` turns E off. `info` prints the current widths and how often each view's "above" matched.

No LSTM, graph, SNN, image, or audio in this port.
