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

Specialists: order 0–4 plus a letter/digit/space/punct class vote, mixed in stretch space. No LSTM, graph, SNN, image, or audio in this port.
