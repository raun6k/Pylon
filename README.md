# Pylon

Pylon is a single-GPU inference server for [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). It exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint. The weights stay fixed.

Each step decodes every active request, then prefills one 256-token chunk of a waiting prompt. If the decode batch is already full and the oldest unfinished prompt has been waiting less than 100 ms, prefill waits. Time to first token can increase in that case. KV is stored in 256-token pages. A full page of prompt tokens is recorded as soon as it is resident, and that page is kept once the same block has been seen again. Set `PYLON_PREFIX_CACHE=false` to disable the prefix cache.

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

## License

Apache-2.0. See `LICENSE`.
