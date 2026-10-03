# Pylon

Pylon is a single-GPU inference server for [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) with an OpenAI-compatible `POST /v1/chat/completions` endpoint, and the weights stay fixed.

On one NVIDIA L4, CUDA-graph decode raised output speed from 24.8 to 26.2 tokens/s at concurrency 1, and from 176.0 to 184.9 at concurrency 8, versus this server's own eager decode. Speculation, prefix cache, and admit-skip were off. Temperature was 0.

<p><img src="benchmarks/plots/decode_heavy_output_tokens_per_second.png" alt="decode_heavy output tokens per second" width="46%">&nbsp;&nbsp;&nbsp;&nbsp;<img src="benchmarks/plots/decode_heavy_inter_token_p50.png" alt="decode_heavy inter-token p50" width="46%"></p>

## Decode benchmark

| Concurrency | CUDA graphs | Output tokens/s | TTFT p50 (ms) | TTFT p99 (ms) | Inter-token p50 (ms) | Peak GPU memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | off | 24.8 | 51.1 | 51.8 | 40.3 | 19.77 GiB |
| 1 | on | 26.2 | 51.6 | 52.3 | 38.1 | 19.94 GiB |
| 8 | off | 176.0 | 237.7 | 695.5 | 44.4 | 19.77 GiB |
| 8 | on | 184.9 | 233.4 | 678.6 | 42.3 | 19.94 GiB |

NVIDIA L4, Qwen/Qwen3-4B-Instruct-2507 snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`, PyTorch 2.14.1+cu130, CUDA 13.0, driver 595.58.03, batch cap 8, prefill chunk 256, 32 requests, 3 repeats. `eager` means graphs off and `pylon` means graphs on.

## What it does

- Decode-first waves
- CUDA-graph decode
- Prefix cache
- Prompt n-gram verify
- Skip-ahead admission

Each step decodes every active request. It then prefills one 256-token chunk of a waiting prompt. If the decode batch is already full and the oldest unfinished prompt has been waiting less than 100 ms, prefill waits. Time to first token can increase in that case. KV is stored in 256-token pages. A full page of prompt tokens is recorded as soon as it is resident. That page is kept once the same block has been seen again. Set `PYLON_PREFIX_CACHE=false` to disable the prefix cache. Each admitted request reserves its whole prompt and its full generation, rounded up to pages. One free page is kept for every request already decoding. If the first waiting request does not fit, a later request that does can start. `PYLON_ADMIT_SKIP` is how many waiting requests may be passed over. It defaults to 4. A request that has waited 100 ms stays next.

## Requirements

- An NVIDIA GPU with CUDA. The server exits if CUDA is missing. An L4 (24 GB) is the reference card.
- Python 3.11 or newer, with PyTorch 2.13 or newer.

## Run

```bash
uv run pylon
```

The server listens on `127.0.0.1:8000`. `GET /health` is ready after the weights load and warmup finishes.

## Unit tests

Unit tests, including the ones that run without a GPU:

```bash
uv run python -m unittest discover -s tests
```

## Smoke check

```bash
uv run python benchmarks/run.py \
  --systems eager \
  --workload decode_heavy \
  --concurrency 1 \
  --limit 8 \
  --repeats 1 \
  --label eager-decode-heavy-c1
```

This runs against a server that is already up. It is not the published result.

## Reproduce

Needs an NVIDIA GPU with CUDA, Python 3.11+, and PyTorch 2.13+.

```bash
uv run python benchmarks/run.py \
  --systems eager,pylon \
  --workload decode_heavy \
  --concurrency 1,8 \
  --repeats 3 \
  --label graph-decode-heavy
```

`eager` is graphs off. `pylon` is graphs on. Both leave speculation, the prefix cache, and admit-skip off. This writes the table above. It does not compare against another server.

## License

Apache-2.0. See `LICENSE`.
