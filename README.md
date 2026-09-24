# bitmoodmouth

Generator (mouth) extracted from [jdnitrap/bitmood](https://github.com/jdnitrap/bitmood).

This repo is a copy of the **bit/byte generation** path only: `generate`, `write`, plus `train` / `info` / `graph` so a memory can be built and used. Compression (`compress` / `decompress`), `demo`, and `compare` stay in bitmood.

The predictor engine under `src/model` is included because generation samples from those bit predictions. It is not a rewrite.

```
make
./bitmoodmouth train --state brain.bin notes.txt
./bitmoodmouth generate 300 "The " --state brain.bin --temp 0.8 --seed 42
./bitmoodmouth write --state brain.bin --out draft.txt
```

Source snapshot taken from `jdnitrap/bitmood` at commit `4818218972b391860d6a77f9ea626d0a428763d3`.
