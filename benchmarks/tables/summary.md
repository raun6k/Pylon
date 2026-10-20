# Benchmark summary

GPU NVIDIA L4, driver 580.126.20, PyTorch 2.14.1+cu130, CUDA 13.0, snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`.
- eager: batch cap 32, prefill chunk 2048.
- pylon (before): batch cap 8, prefill chunk 256.
- pylon: batch cap 32, prefill chunk 2048.

Rows come from these files in `benchmarks/results/`:

- eager: decode_heavy, prefill_heavy, shared_prefix from `20261020T123518Z-full-eager.json`
- pylon (before): decode_heavy from `20261020T112233Z-main-pylon-decode.json`; prefill_heavy, shared_prefix from `20261020T113438Z-main-pylon-prefill.json`
- pylon: decode_heavy, shared_prefix from `20261020T103106Z-full-pylon.json`; prefill_heavy from `20261020T114518Z-pylon-prefill-fixed.json`
- vLLM 0.30.0: decode_heavy, prefill_heavy, shared_prefix from `20261020T111315Z-full-vllm.json`

TTFT is server time from the request joining the queue to its first token. 32 requests, 3 repeats, temperature 0. pylon and vLLM run with the prefix cache on for shared_prefix only; eager never has it.

### decode_heavy

| Concurrency | Metric | eager | pylon (before) | pylon | vLLM 0.30.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Output tokens/s | 26.7 | – | 27.7 | 30.1 |
|  | TTFT p50 / p99 (ms) | 49 / 50 | – | 50 / 51 | 42 / 43 |
|  | Inter-token p50 / p99 (ms) | 37 / 37 | – | 36 / 36 | 33 / 33 |
|  | Prefill tokens computed | 12,288 | – | 12,288 | 12,288 |
| 8 | Output tokens/s | 194.0 | 184.0 | 199.8 | 219.8 |
|  | TTFT p50 / p99 (ms) | 201 / 251 | 235 / 687 | 189 / 262 | 173 / 199 |
|  | Inter-token p50 / p99 (ms) | 40 / 41 | 42 / 43 | 39 / 40 | 36 / 36 |
|  | Prefill tokens computed | 12,288 | 12,288 | 12,288 | 12,288 |
| 16 | Output tokens/s | 357.5 | 183.9 | 367.0 | 412.5 |
|  | TTFT p50 / p99 (ms) | 399 / 472 | 21,883 / 22,535 | 422 / 474 | 315 / 337 |
|  | Inter-token p50 / p99 (ms) | 43 / 44 | 42 / 43 | 42 / 43 | 38 / 38 |
|  | Prefill tokens computed | 12,288 | 12,288 | 12,288 | 12,288 |

### prefill_heavy

| Concurrency | Metric | eager | pylon (before) | pylon | vLLM 0.30.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Output tokens/s | 19.9 | 17.4 | 20.2 | 22.9 |
|  | TTFT p50 / p99 (ms) | 378 / 383 | 574 / 581 | 386 / 394 | 326 / 331 |
|  | Inter-token p50 / p99 (ms) | 39 / 39 | 40 / 41 | 38 / 38 | 34 / 34 |
|  | Prefill tokens computed | 196,608 | 196,608 | 196,608 | 196,608 |
| 8 | Output tokens/s | 54.6 | 35.5 | 53.7 | 64.1 |
|  | TTFT p50 / p99 (ms) | 820 / 3,232 | 4,824 / 5,370 | 847 / 3,348 | 1,025 / 2,769 |
|  | Inter-token p50 / p99 (ms) | 122 / 134 | 72 / 113 | 124 / 136 | 83 / 112 |
|  | Prefill tokens computed | 196,608 | 196,608 | 196,608 | 196,608 |
| 16 | Output tokens/s | 61.7 | 39.0 | 59.6 | 72.6 |
|  | TTFT p50 / p99 (ms) | 853 / 6,581 | 11,028 / 11,424 | 893 / 6,906 | 1,080 / 5,621 |
|  | Inter-token p50 / p99 (ms) | 224 / 242 | 64 / 68 | 234 / 251 | 173 / 203 |
|  | Prefill tokens computed | 196,608 | 196,608 | 196,608 | 196,608 |

### shared_prefix

| Concurrency | Metric | eager | pylon (before) | pylon | vLLM 0.30.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Output tokens/s | 25.6 | 25.6 | 27.2 | 29.8 |
|  | TTFT p50 / p99 (ms) | 120 / 123 | 53 / 187 | 51 / 121 | 37 / 46 |
|  | Inter-token p50 / p99 (ms) | 38 / 38 | 39 / 39 | 36 / 36 | 33 / 33 |
|  | Prefill tokens computed | 61,440 | 13,312 | 13,312 | 5,632 |
| 8 | Output tokens/s | 140.5 | 153.2 | 179.2 | 219.1 |
|  | TTFT p50 / p99 (ms) | 600 / 985 | 253 / 721 | 245 / 272 | 71 / 84 |
|  | Inter-token p50 / p99 (ms) | 48 / 54 | 48 / 50 | 41 / 43 | 35 / 36 |
|  | Prefill tokens computed | 61,440 | 12,288 | 12,288 | 1,536 |
| 16 | Output tokens/s | 208.5 | 154.4 | 303.0 | 410.0 |
|  | TTFT p50 / p99 (ms) | 742 / 1,969 | 3,363 / 3,945 | 453 / 505 | 60 / 82 |
|  | Inter-token p50 / p99 (ms) | 63 / 71 | 49 / 50 | 45 / 50 | 37 / 38 |
|  | Prefill tokens computed | 61,440 | 12,288 | 12,288 | 1,536 |

### pylon vs pylon (before)

| Workload | Concurrency | Output tokens/s | TTFT p50 | Inter-token p50 |
| --- | ---: | ---: | ---: | ---: |
| decode_heavy | 8 | 1.09x | 0.81x | 0.93x |
| decode_heavy | 16 | 2.00x | 0.02x | 0.99x |
| prefill_heavy | 1 | 1.16x | 0.67x | 0.95x |
| prefill_heavy | 8 | 1.51x | 0.18x | 1.72x |
| prefill_heavy | 16 | 1.53x | 0.08x | 3.66x |
| shared_prefix | 1 | 1.06x | 0.97x | 0.94x |
| shared_prefix | 8 | 1.17x | 0.97x | 0.85x |
| shared_prefix | 16 | 1.96x | 0.13x | 0.92x |

### pylon vs vLLM 0.30.0

| Workload | Concurrency | Output tokens/s | TTFT p50 | Inter-token p50 |
| --- | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | 0.92x | 1.20x | 1.09x |
| decode_heavy | 8 | 0.91x | 1.09x | 1.10x |
| decode_heavy | 16 | 0.89x | 1.34x | 1.11x |
| prefill_heavy | 1 | 0.88x | 1.18x | 1.12x |
| prefill_heavy | 8 | 0.84x | 0.83x | 1.50x |
| prefill_heavy | 16 | 0.82x | 0.83x | 1.35x |
| shared_prefix | 1 | 0.91x | 1.40x | 1.09x |
| shared_prefix | 8 | 0.82x | 3.46x | 1.15x |
| shared_prefix | 16 | 0.74x | 7.49x | 1.19x |
