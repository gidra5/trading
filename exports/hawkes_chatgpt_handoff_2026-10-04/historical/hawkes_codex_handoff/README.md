# BTCUSDT Hawkes / latent-channel research handoff

This directory packages the current state of the BTCUSDT spot aggregate-trade study for continuation in Codex/local compute.

Start with:

1. `CODEX_TASK.md` — concise execution brief for Codex.
2. `SPEC.md` — full model, experimental protocol, known results, and caveats.
3. `experiments/` — cleaned reference versions of the latest screening scripts used in ChatGPT.
4. `results/` — CSV outputs from the latest runs.
5. `reports/` — selected earlier HTML reports for context.

The large raw dataset is intentionally not bundled. Put it at, for example:

`data/BTCUSDT-aggTrades-live-2026-08-17_2026-08-30.jsonl.gz`

The raw file has fields:

- `timestampMs`
- `price`
- `quantity`
- `aggressorSide`

Important: equal-millisecond records are in true arrival order. Preserve that order for continuous-time/event-level work.

The scripts here are **references**, not the desired final implementation. In particular, the latent-channel screening scripts infer a residual-based proxy and must not be mistaken for a proper latent point-process fit.
