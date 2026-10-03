import argparse
import inspect
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import benchmarks.run as harness
from benchmarks.run import (
    VLLM_ARGV,
    VLLM_VERSION,
    decode_heavy_series,
    descendant_pids,
    draw_comparison,
    load_workload,
    metrics_from_response,
    ordered_lines,
    parse_used_gpu_memory,
    percentile,
    query_gpu_identity,
    result_flags,
    reuse_running_server,
    should_write_plots,
    stop_process,
    summarize,
    system_environ,
    write_comparison_plots,
)

ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def test_percentile_interpolates_and_a_single_sample_is_both_p50_and_p99(self) -> None:
        self.assertEqual(percentile([3.0], 0.50), 3.0)
        self.assertEqual(percentile([3.0], 0.99), 3.0)
        self.assertEqual(percentile([0.0, 10.0], 0.50), 5.0)
        self.assertEqual(percentile([0.0, 10.0, 20.0], 0.50), 10.0)

    def test_workloads_match_the_benchmark_table_without_a_socket(self) -> None:
        expected = {
            "decode_heavy": (128, 512),
            "prefill_heavy": (2048, 32),
            "shared_prefix": (640, 64),
        }
        for name, (prompt_tokens, max_tokens) in expected.items():
            with self.subTest(workload=name):
                rows = load_workload(ROOT / "benchmarks" / "workloads" / f"{name}.json")
                self.assertEqual(len(rows), 32)
                self.assertTrue(all(row.prompt_tokens == prompt_tokens for row in rows))
                self.assertTrue(all(row.max_tokens == max_tokens for row in rows))
                self.assertTrue(all(row.temperature == 0 for row in rows))
                self.assertTrue(all(row.top_p == 1 for row in rows))
                contents = [message.content for row in rows for message in row.messages]
                self.assertTrue(contents)
                self.assertTrue(all(1 <= len(content) <= 20_000 for content in contents))
        shared = load_workload(ROOT / "benchmarks" / "workloads" / "shared_prefix.json")
        prefixes = {row.messages[0].content[:200] for row in shared}
        self.assertEqual(len(prefixes), 1)
        decode = load_workload(ROOT / "benchmarks" / "workloads" / "decode_heavy.json")
        self.assertEqual(len({row.messages[0].content for row in decode}), 32)

    def test_harness_does_not_import_vllm(self) -> None:
        source = inspect.getsource(harness)
        self.assertNotIn("import vllm", source)
        self.assertNotIn("from vllm", source)
        self.assertNotIn("vllm", sys.modules)

    def test_published_intervals_use_one_definition_for_each_column(self) -> None:
        server = metrics_from_response(
            {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "timings": {
                    "time_to_first_token_seconds": 1.25,
                    "tokenize_seconds": 0.25,
                    "inter_token_seconds": [0.1, 0.3],
                },
            }
        )
        other = metrics_from_response(
            {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
                "timings": {
                    "time_to_first_token_seconds": 2.0,
                    "tokenize_seconds": 0.0,
                    "inter_token_seconds": [0.4, 0.4],
                },
            }
        )
        single = metrics_from_response(
            {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "timings": {
                    "time_to_first_token_seconds": 0.5,
                    "tokenize_seconds": 0.1,
                    "inter_token_seconds": [9.0],
                },
            }
        )
        upstream = metrics_from_response(
            {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "metrics": {
                    "queue_time_ms": 20.0,
                    "time_to_first_token_ms": 80.0,
                    "mean_itl_ms": 9.0,
                },
            }
        )
        quiet = metrics_from_response(
            {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "metrics": {
                    "queue_time_ms": 5.0,
                    "time_to_first_token_ms": 5.0,
                    "mean_itl_ms": None,
                },
            }
        )
        self.assertEqual(server["ttft"], 1.0)
        self.assertEqual(server["inter_token"], 0.2)
        self.assertEqual(server["gaps"], [0.1, 0.3])
        self.assertEqual(upstream["ttft"], 100.0 / 1000)
        self.assertEqual(upstream["inter_token"], 9.0 / 1000)
        self.assertEqual(upstream["gaps"], [])
        self.assertIsNone(upstream["accepted_tokens_per_step"])
        self.assertIsNone(single["inter_token"])
        self.assertEqual(single["gaps"], [9.0])
        self.assertIsNone(quiet["inter_token"])
        summary = summarize([server, other, single, upstream, quiet], elapsed_seconds=2.0)
        self.assertEqual(summary["request_count"], 5)
        self.assertEqual(
            summary["ttft_seconds"]["p50"],
            percentile([1.0, 2.0, 0.4, 100.0 / 1000, 10.0 / 1000], 0.50),
        )
        self.assertEqual(
            summary["inter_token_seconds"]["p50"],
            percentile([0.2, 0.4, 9.0 / 1000], 0.50),
        )
        self.assertEqual(
            summary["inter_token_seconds"]["p99"],
            percentile([0.2, 0.4, 9.0 / 1000], 0.99),
        )
        self.assertNotEqual(
            summary["inter_token_seconds"]["p50"],
            percentile([0.1, 0.3, 0.4, 0.4], 0.50),
        )
        self.assertEqual(summary["prefill_tokens_computed"], 8 + 6 + 8 + 8 + 8)
        self.assertEqual(summary["completion_tokens"], 3 + 3 + 1 + 4 + 1)
        self.assertEqual(summary["output_tokens_per_second"], 12 / 2.0)
        self.assertIsNone(summary["mean_accepted_tokens"])
        with self.assertRaises(RuntimeError):
            metrics_from_response(
                {
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                }
            )
        with self.assertRaises(RuntimeError):
            metrics_from_response(
                {
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                    "metrics": {
                        "queue_time_ms": 1.0,
                        "time_to_first_token_ms": 1.0,
                        "mean_itl_ms": 1.0,
                    },
                }
            )

    def test_peak_memory_uses_the_server_process_tree(self) -> None:
        text = "10, 100\n11, 250\n99, 9000\n"
        pids = descendant_pids(10, {10: [11], 11: [12], 1: [99]})
        self.assertEqual(pids, {10, 11, 12})
        self.assertEqual(parse_used_gpu_memory(text, pids), 250 * 1024 * 1024)
        self.assertEqual(parse_used_gpu_memory(text, None), 9000 * 1024 * 1024)
        self.assertIsNone(parse_used_gpu_memory("not a sample", {10}))

    def test_gpu_identity_does_not_require_a_device(self) -> None:
        identity = query_gpu_identity()
        self.assertEqual(set(identity), {"gpu_name", "driver_version"})

    def test_vllm_argv_is_pinned(self) -> None:
        self.assertEqual(VLLM_VERSION, "0.30.0")
        self.assertEqual(
            VLLM_ARGV[:3],
            ("vllm", "serve", "Qwen/Qwen3-4B-Instruct-2507"),
        )
        text = " ".join(VLLM_ARGV)
        for flag in (
            "--host 127.0.0.1",
            "--port 8001",
            "--dtype bfloat16",
            "--tensor-parallel-size 1",
            "--max-model-len 4096",
            "--gpu-memory-utilization 0.90",
            "--enable-prefix-caching",
            "--enable-per-request-metrics",
            "--enable-prompt-tokens-details",
        ):
            self.assertIn(flag, text)
        self.assertNotIn("--enforce-eager", VLLM_ARGV)
        self.assertNotIn("--max-num-seqs", VLLM_ARGV)
        self.assertFalse(any("draft" in arg or "speculative" in arg for arg in VLLM_ARGV))
        self.assertFalse(reuse_running_server(["vllm"], "vllm"))
        self.assertTrue(reuse_running_server(["eager"], "eager"))
        self.assertFalse(reuse_running_server(["eager", "pylon"], "eager"))

    def test_pylon_column_enables_graphs_and_leaves_eager_flags_off(self) -> None:
        env = system_environ({}, "pylon")
        self.assertEqual(env["PYLON_CUDA_GRAPHS"], "true")
        self.assertEqual(env["PYLON_PREFIX_CACHE"], "false")
        self.assertEqual(env["PYLON_SPECULATE_K"], "1")
        self.assertEqual(env["PYLON_ADMIT_SKIP"], "0")
        self.assertTrue(result_flags("pylon", env)["cuda_graphs"])
        self.assertFalse(result_flags("pylon", env)["speculation"])
        self.assertEqual(result_flags("pylon", env)["admit_skip"], 0)
        self.assertFalse(result_flags("pylon", env)["prefix_cache"])
        self.assertFalse(result_flags("vllm", None)["speculation"])
        self.assertTrue(result_flags("vllm", None)["prefix_cache"])

    def test_plots_use_decode_heavy_p50_and_throughput(self) -> None:
        records = []
        for system, ttft, rate in (("vllm", 0.2, 30.0), ("eager", 1.0, 10.0), ("pylon", 0.5, 20.0)):
            for concurrency in (16, 1, 8):
                records.append(
                    {
                        "system": system,
                        "workload": "decode_heavy",
                        "concurrency": concurrency,
                        "summary": {
                            "ttft_seconds": {"p50": ttft, "p99": ttft + 1},
                            "output_tokens_per_second": rate,
                        },
                    }
                )
        records.append(
            {
                "system": "eager",
                "workload": "prefill_heavy",
                "concurrency": 1,
                "summary": {
                    "ttft_seconds": {"p50": 99.0, "p99": 99.0},
                    "output_tokens_per_second": 1.0,
                },
            }
        )
        series = decode_heavy_series(records)
        self.assertEqual(ordered_lines(series["ttft"]), ["eager", "pylon", "vllm"])
        self.assertEqual(sorted(series["ttft"]["eager"]), [(1, 1.0), (8, 1.0), (16, 1.0)])
        self.assertTrue(
            all(value != 99.0 for points in series["ttft"].values() for _, value in points)
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        try:
            draw_comparison(series["ttft"], "Time to first token p50 (seconds)", axis)
            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            self.assertEqual(labels, ["eager", "pylon", "vllm"])
            self.assertEqual(axis.get_ylabel(), "Time to first token p50 (seconds)")
            self.assertEqual(axis.get_xlabel(), "Concurrency")
            self.assertEqual([float(tick) for tick in axis.get_xticks()], [1.0, 8.0, 16.0])
        finally:
            plt.close(figure)
        with tempfile.TemporaryDirectory() as directory:
            written = write_comparison_plots(records, Path(directory))
            self.assertEqual(
                sorted(path.name for path in written),
                [
                    "output_tokens_per_second_vs_concurrency.png",
                    "ttft_vs_concurrency.png",
                ],
            )
            for path in written:
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
        full = argparse.Namespace(limit=0, repeats=3)
        self.assertTrue(should_write_plots(full, records))
        self.assertFalse(should_write_plots(argparse.Namespace(limit=8, repeats=3), records))
        self.assertFalse(should_write_plots(argparse.Namespace(limit=0, repeats=1), records))
        without_vllm = [record for record in records if record["system"] != "vllm"]
        self.assertFalse(should_write_plots(full, without_vllm))

    def test_stop_process_kills_the_spawned_group(self) -> None:
        process = subprocess.Popen(["sleep", "30"], start_new_session=True)
        try:
            stop_process(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
