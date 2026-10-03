# Pylon

Pylon is a single-GPU inference server for [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). It exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint. The weights stay fixed.

Each step decodes every active request, then prefills one 256-token chunk of a waiting prompt. If the decode batch is already full and the oldest unfinished prompt has been waiting less than 100 ms, prefill waits. Time to first token can increase in that case. KV is stored in 256-token pages. A full page of prompt tokens is recorded as soon as it is resident, and that page is kept once the same block has been seen again. Set `PYLON_PREFIX_CACHE=false` to disable the prefix cache. Each admitted request reserves its whole prompt and its full generation, rounded up to pages, and one free page is kept for every request already decoding. If the first waiting request does not fit, a later request that does can start. `PYLON_ADMIT_SKIP` is how many waiting requests may be passed over and defaults to 4. A request that has waited 100 ms stays next.

## Requirements

- An NVIDIA GPU with CUDA. The server exits if CUDA is missing. An L4 (24 GB) is the reference card.
- Python 3.11 or newer, with PyTorch 2.13 or newer.

## Run

```bash
uv run pylon
```

The server listens on `127.0.0.1:8000`. `GET /health` is ready after the weights load and warmup finishes.

Unit tests, including the ones that run without a GPU:

```bash
uv run python -m unittest discover -s tests
```

A short pipeline check against a server that is already up:

```bash
uv run python benchmarks/run.py \
  --systems eager \
  --workload decode_heavy \
  --concurrency 1 \
  --limit 8 \
  --repeats 1 \
  --label eager-decode-heavy-c1
```

That command is a pipeline check. It is not a published latency result. Tables in this README are written by the benchmark harness. A number is not added by hand.

The harness can also write `benchmarks/plots/ttft_vs_concurrency.png` and `benchmarks/plots/output_tokens_per_second_vs_concurrency.png`. This README does not include those two plots.

## Decode benchmark

`decode_heavy` at concurrency 1 and 8, on one NVIDIA L4, with weights `Qwen/Qwen3-4B-Instruct-2507` and temperature 0. Speculation and admit-skip were off in both columns. The prefix cache was off in both columns. The only difference is CUDA-graph replay for decode. The gain is small versus this server's own eager decode.

The harness wrote `benchmarks/results/20261003T153549Z-graph-decode-heavy.json` from 32 requests and 3 repeats. Driver 595.58.03, PyTorch 2.14.1+cu130, CUDA 13.0, batch cap 8, prefill chunk size 256. Weight snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`.

```bash
python benchmarks/run.py \
  --systems eager,pylon \
  --workload decode_heavy \
  --concurrency 1,8 \
  --repeats 3 \
  --label graph-decode-heavy
```

In that command, `eager` is graphs off and `pylon` is graphs on.

| Concurrency | CUDA graphs | Output tokens/s | TTFT p50 (s) | TTFT p99 (s) | Inter-token p50 (s) | Peak GPU memory (bytes) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | off | 24.7970 | 0.0511 | 0.0518 | 0.0403 | 21225275392 |
| 1 | on | 26.1710 | 0.0516 | 0.0523 | 0.0381 | 21407727616 |
| 8 | off | 176.0118 | 0.2377 | 0.6955 | 0.0444 | 21225275392 |
| 8 | on | 184.8867 | 0.2334 | 0.6786 | 0.0423 | 21407727616 |

Output tokens per second is higher with graphs on, and inter-token p50 is lower, at both concurrencies. At concurrency 1, time to first token is higher with graphs on. At concurrency 8, time to first token is lower with graphs on. Peak GPU memory is 21407727616 bytes with graphs on and 21225275392 bytes with graphs off.

![decode_heavy output tokens per second](benchmarks/plots/decode_heavy_output_tokens_per_second.png)

![decode_heavy inter-token p50](benchmarks/plots/decode_heavy_inter_token_p50.png)

The inter-token chart uses the same p50 in milliseconds: 40.3 and 38.1 at concurrency 1, 44.4 and 42.3 at concurrency 8.

## License

Apache-2.0. See `LICENSE`.
