# Storage contract

The repository has two physical content-addressed stores plus explicit mutable
training, market, and backend workspaces. Runtime code reads this layout directly;
there are no aliases, materialized compatibility files, or legacy-path fallbacks.

```text
data/
  market/
    immutable/
      objects/sha256/<prefix>/<hash>.zst
      refs/candles/<market>/<symbol>/<interval>/<date>.json
    mutable/
      candles/<market>/<symbol>/<interval>/<date>.jsonl
      streams/<market>/<symbol>-orderbook.jsonl
      tmp/
  training/
    immutable/
      objects/sha256/<prefix>/<hash>.{zst,bin}
      refs/features/...
      refs/oracle/{1s,1m}/...
      refs/training/...
    datasets/<dataset-id>/
      dataset.json
      components/**/*.json
      state/progress.json
    runs/<run-id>/
      checkpoints/{last,best}.json
      logs/
        training.history.jsonl
        training.jsonl
      state/
      control/
      tmp/
    cache/{torchinductor,triton,cuda,tmp}/
    analysis/
  runtime/
    state/
    backtests/
    logs/
```

`data/models` is reserved for verified serving artifacts such as ONNX models. It
does not contain resumable training checkpoints.

## Immutable data

Market candles, input features, one-second and one-minute oracle distributions,
teacher arrays, dataset pair arrays, and checkpoints are addressed by SHA-256.
The hash is calculated over the canonical uncompressed bytes. Equal payloads
therefore occupy one object even when several datasets refer to them.

Sequential tensor references record an implicit axis:

```text
timestamp(row) = sequence.start + row * sequence.step + cumulative sparse time jumps
```

This removes repeated timestamp arrays. Complete one-second UTC days use 86,400
rows; completed-minute paths use 1,441 rows including the minute immediately
before the day. Tensor payloads are row-major typed bytes compressed with
Zstandard.

Candle payloads are columnar. Constant symbol, interval, closed state, and close
offset are stored once; OHLCV values use exact decimal scaling, delta coding,
zig-zag varints, and Zstandard. Normal timestamps stay implicit; a candle gap adds
one compact time-jump entry instead of a full timestamp column. A complete empty
day is a zero-row shard. High and low are retained, so canonical candle storage
includes wicks.

PyTorch checkpoints use uncompressed `.bin` objects because `torch.load` can read
them directly without another full-size temporary file or a large decompression
buffer. Space is bounded by pointer retention and orphan collection instead of
making checkpoint resume slower and less memory-efficient.

## Mutable data

A dataset directory owns only manifests, progress state, and direct JSON
references to immutable objects. It never owns a second copy or hard link of a
large tensor.

The server records closed live candles through the same market store used by
historical ingestion. The active UTC day is the only mutable candle payload and
is split into a date-scoped JSONL staging file. Once every expected interval is
present, the recorder atomically publishes the day as a compressed immutable
columnar shard and removes the staging file. Multi-day intervals publish each
closed candle as its own date-keyed shard. Reads combine immutable daily
references with the current partial day, preferring the immutable reference if
historical ingestion seals that day first. Order-book snapshots remain mutable
streams because they are not candle history.

A run directory owns mutable status, append-only logs, controls, and checkpoint
pointers. `logs/training.jsonl` is the active writer-owned metric segment. A
resumed or migrated run keeps its closed, write-once segment as
`logs/training.history.jsonl`; readers concatenate history and active metrics in
that order and expose one cursor. Updating `checkpoints/last.json` atomically
advances the pointer to a new immutable checkpoint object. `best.json` can retain
a different object.
Objects no longer reachable from any pointer are collected after a one-hour
commit-race grace period during subsequent checkpoint writes and training
launcher maintenance.

Compiler caches are explicitly disposable. The launcher places every ML cache
under `data/training/cache`, caps it at 12 GiB by default, preserves 32 GiB of
free disk when possible, and prunes again after the child process exits. Override
those boundaries with `TRADING_ML_CACHE_MAX_GIB`,
`TRADING_DISK_RESERVE_GIB`, and `TRADING_STORAGE_ORPHAN_GRACE_HOURS`.
Checkpoint writers throttle full reference scans to once every 15 minutes by
default; `TRADING_STORAGE_GC_INTERVAL_MINUTES` adjusts that interval.

## Audit and retention

```sh
npm run storage:audit
npm run storage:gc
npm run storage:gc -- --apply --minimum-age-hours 1
npm run storage:dedupe -- --root data/ml-datasets
npm run storage:dedupe -- --root data/ml-datasets --apply --confirm-no-writers
```

Audit follows references in both immutable catalogs and mutable dataset/run
trees. Garbage collection is a dry run unless `--apply` is supplied. Invalid
references are reported and never treated as permission to remove their
objects.

The deduplication command is for pre-migration or externally supplied directory
trees that still contain large payload files. It groups same-name/same-size
candidates, verifies every match with SHA-256, recognizes existing hard links by
filesystem identity, and atomically replaces only byte-identical redundant files
with hard links. It is a dry run unless `--apply` and `--confirm-no-writers` are
both supplied. Canonical immutable stores already deduplicate by content hash and
do not need this pass.

## One-way migration

The runtime does not understand `data/historical`, `data/ml-datasets`,
`data/ml-runs`, `data/ml-*-studies`, `data/ml-analysis`, or `data/runtime-cache`.
A dedicated one-way migration converts those formats and then removes the old
payload files:

```sh
npm run storage:migrate -- --all
npm run storage:migrate -- --all --apply --confirm-stopped
npm run storage:migrate -- --live-candles --apply --confirm-stopped
```

The first command reports scope without changing disk. Applying requires an
explicit confirmation that all legacy writers and trainers are stopped. The
migrator converts candles, adopts dataset tensors into immutable storage,
replaces stored timestamp arrays with implicit axes, adopts raw `.pt` files from
both runs and model export directories into checkpoint pointers, moves
logs/status/control files into run subdirectories, and moves regenerable caches
under `training/cache`. Backend state, backtests, process logs, and mutable market
streams move under `runtime/` and `market/mutable/`. Existing unbounded live
candle streams are split by UTC date: complete days become immutable shards and
only incomplete days remain in mutable candle staging.

`--live-candles` scopes the one-way conversion to the old server candle stream,
so only the server candle writer must be stopped; active trainers do not block
that migration.

For a single legacy dataset, use `--dataset data/ml-datasets/<dataset-id>`.
Migration renames the dataset into `data/training/datasets` before converting
one payload at a time, keeping transient disk growth bounded to one component.
