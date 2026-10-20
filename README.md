# Pylon

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements)
[![PyTorch](https://img.shields.io/badge/pytorch-2.13%2B-ee4c2c)](#requirements)

Pylon is a single-GPU inference server for [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). The weights stay fixed. It exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint.

The repository follows one server through three stages and measures each against vLLM 0.30.0 on the same NVIDIA L4:

1. **pylon (before)**, the previous release.
2. **pylon, graphs and cache off**, the current scheduling and memory work on its own: packed decode and prefill ticks, a batch cap of 32, fused projections, and a correctly sized KV pool.
3. **pylon**, the same with CUDA-graph decode, the two-hit prefix cache, and skip-ahead admission turned on.

Most of the gain comes from stage 2. At concurrency 16 it lifts `decode_heavy` from 45% to 87% of vLLM's output rate. Stage 3 adds about 3% on decode and much more when prompts share a prefix (`shared_prefix` at concurrency 16 goes from 51% to 74% of vLLM). Across the three workloads, Pylon reaches 74% to 92% of vLLM's output rate and has the lower median time to first token on long prompts at concurrency 8 and 16.

## Contents

- [What it does](#what-it-does)
- [Benchmark](#benchmark)
- [Requirements](#requirements)
- [Run](#run)
- [Unit tests](#unit-tests)
- [Smoke check](#smoke-check)
- [Reproduce](#reproduce)
- [License](#license)

## What it does

- **Packed ticks.** Each step runs one forward: one token for every decoding request, plus up to `PYLON_PREFILL_CHUNK_SIZE` (default 2048) prompt tokens taken from as many waiting prompts as fit. Only rows that sample a token run the vocabulary head. If the batch is full and the oldest prompt has waited under 100 ms, the step decodes only. A long prompt in a step makes that step's inter-token gap longer.
- **CUDA-graph decode.** A step with no prompt work replays a captured decode graph, one per batch size up to `PYLON_MAX_BATCH_SIZE` (default 32).
- **Fused projections.** Q, K, and V share one matrix multiply, as do the gate and up projections. Rotary tables are gathered once per forward.
- **Paged KV and a two-hit prefix cache.** KV lives in 256-token pages. A full page of prompt tokens is kept once the same block shows up again. A request admitted before that block was kept still picks it up before its first prompt chunk. `PYLON_PREFIX_CACHE=false` turns that off.
- **Prompt n-gram speculation.** This provides a verify path for speculated tokens. It was off in the benchmark.
- **Skip-ahead admission.** Each request reserves its prompt and its full generation in pages. If the first waiter does not fit, a later one that does can start. `PYLON_ADMIT_SKIP` defaults to 4. A request that has waited 100 ms stays next.

## Benchmark

NVIDIA L4, driver 580.126.20, PyTorch 2.14.1+cu130, CUDA 13.0, Qwen/Qwen3-4B-Instruct-2507 snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`, vLLM 0.30.0. 32 requests per cell, 3 repeats, temperature 0, non-streaming. Stages 2 and 3 run with batch cap 32 and prefill chunk 2048; stage 1 ran with batch cap 8 and prefill chunk 256.

- **decode_heavy:** 128 prompt tokens, 512 new tokens.
- **prefill_heavy:** 2048 prompt tokens, 32 new tokens.
- **shared_prefix:** 640 prompt tokens, the first 512 identical, 64 new tokens.

<p align="center"><img src="benchmarks/plots/output_tokens_per_second_vs_concurrency.png" alt="decode_heavy output tokens per second for each stage and vLLM at concurrency 1, 8, and 16" width="90%"></p>

<p align="center"><img src="benchmarks/plots/ttft_vs_concurrency.png" alt="decode_heavy time to first token p50 for each stage and vLLM at concurrency 1, 8, and 16" width="90%"></p>

<p align="center"><sub>decode_heavy output tokens per second, then time to first token p50. Percentages are relative to vLLM. Stage 1 was not run at concurrency 1.</sub></p>

### Each stage as a percentage of vLLM

Output tokens per second (higher is better) / time to first token p50 (lower is better). A dash means that row was not measured.

| Workload | Concurrency | pylon (before) | pylon, graphs and cache off | pylon |
| --- | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | – | 89% / 117% | 92% / 120% |
| decode_heavy | 8 | 84% / 135% | 88% / 116% | 91% / 109% |
| decode_heavy | 16 | 45% / 6,942% | 87% / 126% | 89% / 134% |
| prefill_heavy | 1 | 76% / 176% | 87% / 116% | 88% / 118% |
| prefill_heavy | 8 | 55% / 471% | 85% / 80% | 84% / 83% |
| prefill_heavy | 16 | 54% / 1,021% | 85% / 79% | 82% / 83% |
| shared_prefix | 1 | 86% / 145% | 86% / 327% | 91% / 140% |
| shared_prefix | 8 | 70% / 359% | 64% / 849% | 82% / 346% |
| shared_prefix | 16 | 38% / 5,563% | 51% / 1,228% | 74% / 749% |

### Measured values

**Output tokens per second**

| Workload | Concurrency | pylon (before) | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | – | 26.7 | 27.7 | 30.1 |
| decode_heavy | 8 | 184.0 | 194.0 | 199.8 | 219.8 |
| decode_heavy | 16 | 183.9 | 357.5 | 367.0 | 412.5 |
| prefill_heavy | 1 | 17.4 | 19.9 | 20.2 | 22.9 |
| prefill_heavy | 8 | 35.5 | 54.6 | 53.7 | 64.1 |
| prefill_heavy | 16 | 39.0 | 61.7 | 59.6 | 72.6 |
| shared_prefix | 1 | 25.6 | 25.6 | 27.2 | 29.8 |
| shared_prefix | 8 | 153.2 | 140.5 | 179.2 | 219.1 |
| shared_prefix | 16 | 154.4 | 208.5 | 303.0 | 410.0 |

**Time to first token p50 (ms)**

| Workload | Concurrency | pylon (before) | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | – | 49 | 50 | 42 |
| decode_heavy | 8 | 235 | 201 | 189 | 173 |
| decode_heavy | 16 | 21,883 | 399 | 422 | 315 |
| prefill_heavy | 1 | 574 | 378 | 386 | 326 |
| prefill_heavy | 8 | 4,824 | 820 | 847 | 1,025 |
| prefill_heavy | 16 | 11,028 | 853 | 893 | 1,080 |
| shared_prefix | 1 | 53 | 120 | 51 | 37 |
| shared_prefix | 8 | 253 | 600 | 245 | 71 |
| shared_prefix | 16 | 3,363 | 742 | 453 | 60 |

**Inter-token p50 (ms)**

| Workload | Concurrency | pylon (before) | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | – | 37 | 36 | 33 |
| decode_heavy | 8 | 42 | 40 | 39 | 36 |
| decode_heavy | 16 | 42 | 43 | 42 | 38 |
| prefill_heavy | 1 | 40 | 39 | 38 | 34 |
| prefill_heavy | 8 | 72 | 122 | 124 | 83 |
| prefill_heavy | 16 | 64 | 224 | 234 | 173 |
| shared_prefix | 1 | 39 | 38 | 36 | 33 |
| shared_prefix | 8 | 48 | 48 | 41 | 35 |
| shared_prefix | 16 | 49 | 63 | 45 | 37 |

Time to first token is server time from the request joining the queue to the first token. The full tables, with p99 and the prefill tokens each run computed, are in [`benchmarks/tables/summary.md`](benchmarks/tables/summary.md), which names the result file behind every row.

- **Stage 1 to stage 2:** `decode_heavy` at concurrency 16 goes from 183.9 to 357.5 output tokens per second, and `prefill_heavy` median time to first token at concurrency 8 and 16 goes from 4.8 s and 11.0 s to 0.82 s and 0.85 s. Part of this is a startup bug in stage 1 that left most free GPU memory out of the KV pool, so long prompts queued behind a handful of requests. With the fix the pool is 11.1 GiB (`kv_budget_bytes` in the result files).
- **Stage 2 to stage 3:** CUDA graphs raise `decode_heavy` output rate by 3.7%, 3.0%, and 2.6% at concurrency 1, 8, and 16. Decode is limited by reading the weights from memory, so removing launch overhead buys little. The prefix cache is the larger step: on `shared_prefix`, output rate rises 28% at concurrency 8 and 45% at concurrency 16, and median time to first token falls from 600 ms to 245 ms at concurrency 8. On `prefill_heavy`, stage 2 has 2% to 5% lower median time to first token than stage 3 and up to 4% higher output rate; with one run per stage that difference is not resolved.
- **Against vLLM:** inter-token p50 is about 10% higher on `decode_heavy`, and 1.35x to 1.5x higher on `prefill_heavy` at concurrency 8 and 16, because every step carries a 2048-token prefill chunk. On `shared_prefix`, Pylon reuses whole 256-token pages, so it computes 128 prompt tokens per request; vLLM reuses 16-token blocks and, because every repeat replays the same prompts, computes 16. That is most of the time-to-first-token gap on that workload.
- **Measurement notes:** the stage 3 rows for `prefill_heavy` were measured after the KV pool fix; its other rows were measured before it, and no request in them waited more than 140 ms for KV pages. vLLM runs with prefix caching off on `decode_heavy` and `prefill_heavy`, because every repeat replays the same 32 prompts and a cache would measure hits instead of prefill, and on for `shared_prefix`, as stage 3 does.

## Requirements

- An NVIDIA GPU with CUDA. The server exits if CUDA is missing. An L4 (24 GB) is the reference card.
- Python 3.11 or newer, with PyTorch 2.13 or newer.
- For the vLLM column only: `vllm==0.30.0` on `PATH`, in its own environment.

## Run

```bash
uv run pylon
```

The server listens on `127.0.0.1:8000`. `GET /health` is ready after the weights load and warmup finishes.

## Unit tests

To run the unit tests, including those that run without a GPU:

```bash
uv run python -m unittest discover -s tests
```

## Smoke check

<details>
<summary>This runs against a server that is already up. It is not the published result.</summary>

```bash
uv run python benchmarks/run.py \
  --systems eager \
  --workload decode_heavy \
  --concurrency 1 \
  --limit 8 \
  --repeats 1 \
  --label eager-decode-heavy-c1
```

</details>

## Reproduce

Reproduction requires an NVIDIA GPU with CUDA, Python 3.11+, and PyTorch 2.13+. The harness names stage 2 `eager` and stage 3 `pylon`. Each command writes its own file to `benchmarks/results/`:

```bash
for system in eager pylon vllm; do
  uv run python benchmarks/run.py \
    --systems $system \
    --workload decode_heavy,prefill_heavy,shared_prefix \
    --concurrency 1,8,16 \
    --repeats 3 \
    --label full-$system
done
uv run python benchmarks/summarize.py before=<file> eager=<file> pylon=<file> vllm=<file>
```

`eager` turns off graphs, the prefix cache, speculation, and skip-ahead admission. `pylon` turns on graphs and skip-ahead admission and leaves speculation off. The prefix cache follows the workload for `pylon` and `vllm`: off for `decode_heavy` and `prefill_heavy`, on for `shared_prefix`. Stage 1 is the `src/` of commit `6ae4722` run under this harness as `pylon` with `PYLON_MAX_BATCH_SIZE=8` and `PYLON_PREFILL_CHUNK_SIZE=256`. The harness starts one server at a time and stops it before the next. `summarize.py` writes the tables and the two plots.

## License

Apache-2.0. See `LICENSE`.
