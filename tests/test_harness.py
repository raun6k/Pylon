import unittest
from pathlib import Path

from benchmarks.run import load_workload, percentile

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


if __name__ == "__main__":
    unittest.main()
