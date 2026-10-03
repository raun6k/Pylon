# Pylon

Pylon is a single-GPU inference server for [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). The weights stay fixed. The work is the serving path around them: how requests are admitted, how the KV cache is paged and reused, and how decode steps are executed.

It exposes an OpenAI-compatible chat completions endpoint.

## What it optimizes

- Continuous batching, so many requests share one decode loop.
- Chunked prefill, so a long prompt cannot stall decode indefinitely.
- A paged KV cache with 256-token pages.
- Prefix reuse for prompts that share a long head.
- CUDA-graph replay for decode.
- Decode scheduled ahead of prefill.
- Speculative verification from the request's own prompt, with no second model.
- Admission that does not let one long prompt block shorter jobs behind it.

## Benchmarks

One command compares three servers on the same GPU and the same weights: the unoptimized server, Pylon, and vLLM. It reports p50 and p99 time to first token, p50 and p99 inter-token latency, output tokens per second, prefill tokens actually computed, and peak GPU memory. Workloads are short chats, long prompts, and a shared-prefix set, at concurrency 1, 8, and 16.

Tables and plots in this README are produced by that command. A number is not added by hand.

## Layout

The server is split so the scheduler, the cache, the model forward, and the HTTP API can change independently. See the module map once the tree is in place.

## Requirements

- Linux or a cloud NVIDIA GPU. CUDA does not run on Apple Silicon.
- An NVIDIA GPU with enough memory for Qwen3-4B in BF16, plus KV cache. An L4 (24 GB) is the reference card.
- Python environment and PyTorch with CUDA, pinned in the project once the environment file exists.

## Status

The serving design is specified. Implementation follows that design in reviewable slices. The repository should stay runnable and readable at the end of each slice.

## License

Apache-2.0. See `LICENSE`.
