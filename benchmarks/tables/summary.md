# Benchmark summary

GPU NVIDIA L4, driver 580.126.20, PyTorch 2.14.1+cu130, CUDA 13.0, snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`.
- pylon (before): batch cap 8, prefill chunk 256.
- pylon, graphs and cache off: batch cap 32, prefill chunk 2048.
- pylon: batch cap 32, prefill chunk 2048.

Rows come from these files in `benchmarks/results/`:

- pylon (before): decode_heavy from `20261020T112233Z-main-pylon-decode.json`; prefill_heavy, shared_prefix from `20261020T113438Z-main-pylon-prefill.json`
- pylon, graphs and cache off: decode_heavy, prefill_heavy, shared_prefix from `20261020T123518Z-full-eager.json`
- pylon: decode_heavy, shared_prefix from `20261020T103106Z-full-pylon.json`; prefill_heavy from `20261020T114518Z-pylon-prefill-fixed.json`
- vLLM 0.30.0: decode_heavy, prefill_heavy, shared_prefix from `20261020T111315Z-full-vllm.json`

TTFT is server time from the request joining the queue to its first token. 32 requests, 3 repeats, temperature 0. pylon and vLLM run with the prefix cache on for shared_prefix only; the graphs-and-cache-off stage never has it.

### decode_heavy

| Concurrency | Metric | pylon (before) | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Output tokens/s | – | 26.7 | 27.7 | 30.1 |
|  | TTFT p50 / p99 (ms) | – | 49 / 50 | 50 / 51 | 42 / 43 |
|  | Inter-token p50 / p99 (ms) | – | 37 / 37 | 36 / 36 | 33 / 33 |
|  | Prefill tokens computed | – | 12,288 | 12,288 | 12,288 |
| 8 | Output tokens/s | 184.0 | 194.0 | 199.8 | 219.8 |
|  | TTFT p50 / p99 (ms) | 235 / 687 | 201 / 251 | 189 / 262 | 173 / 199 |
|  | Inter-token p50 / p99 (ms) | 42 / 43 | 40 / 41 | 39 / 40 | 36 / 36 |
|  | Prefill tokens computed | 12,288 | 12,288 | 12,288 | 12,288 |
| 16 | Output tokens/s | 183.9 | 357.5 | 367.0 | 412.5 |
|  | TTFT p50 / p99 (ms) | 21,883 / 22,535 | 399 / 472 | 422 / 474 | 315 / 337 |
|  | Inter-token p50 / p99 (ms) | 42 / 43 | 43 / 44 | 42 / 43 | 38 / 38 |
|  | Prefill tokens computed | 12,288 | 12,288 | 12,288 | 12,288 |

### prefill_heavy

| Concurrency | Metric | pylon (before) | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Output tokens/s | 17.4 | 19.9 | 20.2 | 22.9 |
|  | TTFT p50 / p99 (ms) | 574 / 581 | 378 / 383 | 386 / 394 | 326 / 331 |
|  | Inter-token p50 / p99 (ms) | 40 / 41 | 39 / 39 | 38 / 38 | 34 / 34 |
|  | Prefill tokens computed | 196,608 | 196,608 | 196,608 | 196,608 |
| 8 | Output tokens/s | 35.5 | 54.6 | 53.7 | 64.1 |
|  | TTFT p50 / p99 (ms) | 4,824 / 5,370 | 820 / 3,232 | 847 / 3,348 | 1,025 / 2,769 |
|  | Inter-token p50 / p99 (ms) | 72 / 113 | 122 / 134 | 124 / 136 | 83 / 112 |
|  | Prefill tokens computed | 196,608 | 196,608 | 196,608 | 196,608 |
| 16 | Output tokens/s | 39.0 | 61.7 | 59.6 | 72.6 |
|  | TTFT p50 / p99 (ms) | 11,028 / 11,424 | 853 / 6,581 | 893 / 6,906 | 1,080 / 5,621 |
|  | Inter-token p50 / p99 (ms) | 64 / 68 | 224 / 242 | 234 / 251 | 173 / 203 |
|  | Prefill tokens computed | 196,608 | 196,608 | 196,608 | 196,608 |

### shared_prefix

| Concurrency | Metric | pylon (before) | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Output tokens/s | 25.6 | 25.6 | 27.2 | 29.8 |
|  | TTFT p50 / p99 (ms) | 53 / 187 | 120 / 123 | 51 / 121 | 37 / 46 |
|  | Inter-token p50 / p99 (ms) | 39 / 39 | 38 / 38 | 36 / 36 | 33 / 33 |
|  | Prefill tokens computed | 13,312 | 61,440 | 13,312 | 5,632 |
| 8 | Output tokens/s | 153.2 | 140.5 | 179.2 | 219.1 |
|  | TTFT p50 / p99 (ms) | 253 / 721 | 600 / 985 | 245 / 272 | 71 / 84 |
|  | Inter-token p50 / p99 (ms) | 48 / 50 | 48 / 54 | 41 / 43 | 35 / 36 |
|  | Prefill tokens computed | 12,288 | 61,440 | 12,288 | 1,536 |
| 16 | Output tokens/s | 154.4 | 208.5 | 303.0 | 410.0 |
|  | TTFT p50 / p99 (ms) | 3,363 / 3,945 | 742 / 1,969 | 453 / 505 | 60 / 82 |
|  | Inter-token p50 / p99 (ms) | 49 / 50 | 63 / 71 | 45 / 50 | 37 / 38 |
|  | Prefill tokens computed | 12,288 | 61,440 | 12,288 | 1,536 |

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

### Each stage as a percentage of vLLM 0.30.0

Output tokens per second (higher is better) / time to first token p50 (lower is better).

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
