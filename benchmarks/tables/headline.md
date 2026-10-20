**Output tokens per second**

| Workload | Concurrency | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | 26.7 | 27.7 | 30.1 |
| decode_heavy | 8 | 194.0 | 199.8 | 219.8 |
| decode_heavy | 16 | 357.5 | 367.0 | 412.5 |
| prefill_heavy | 1 | 19.9 | 20.2 | 22.9 |
| prefill_heavy | 8 | 54.6 | 53.7 | 64.1 |
| prefill_heavy | 16 | 61.7 | 59.6 | 72.6 |
| shared_prefix | 1 | 25.6 | 27.2 | 29.8 |
| shared_prefix | 8 | 140.5 | 179.2 | 219.1 |
| shared_prefix | 16 | 208.5 | 303.0 | 410.0 |

**Time to first token p50 (ms)**

| Workload | Concurrency | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | 49 | 50 | 42 |
| decode_heavy | 8 | 201 | 189 | 173 |
| decode_heavy | 16 | 399 | 422 | 315 |
| prefill_heavy | 1 | 378 | 386 | 326 |
| prefill_heavy | 8 | 820 | 847 | 1,025 |
| prefill_heavy | 16 | 853 | 893 | 1,080 |
| shared_prefix | 1 | 120 | 51 | 37 |
| shared_prefix | 8 | 600 | 245 | 71 |
| shared_prefix | 16 | 742 | 453 | 60 |

**Inter-token p50 (ms)**

| Workload | Concurrency | pylon, graphs and cache off | pylon | vLLM 0.30.0 |
| --- | ---: | ---: | ---: | ---: |
| decode_heavy | 1 | 37 | 36 | 33 |
| decode_heavy | 8 | 40 | 39 | 36 |
| decode_heavy | 16 | 43 | 42 | 38 |
| prefill_heavy | 1 | 39 | 38 | 34 |
| prefill_heavy | 8 | 122 | 124 | 83 |
| prefill_heavy | 16 | 224 | 234 | 173 |
| shared_prefix | 1 | 38 | 36 | 33 |
| shared_prefix | 8 | 48 | 41 | 35 |
| shared_prefix | 16 | 63 | 45 | 37 |
