import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from pylon.model.config import MODEL_ID, read_model_config
from pylon.model.tokenizer import read_stop_token_ids, render_chat

FIXTURE = Path(__file__).parent / "fixtures" / "config.json"
ROOT = Path(__file__).resolve().parents[1]

CHAT_TEMPLATE = json.loads(
    (FIXTURE.parent / "tokenizer_config.json").read_text()
)["chat_template"]

GENERATION_CONFIG = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "transformers_version": "4.51.0",
}


class LoaderTests(unittest.TestCase):
    def test_rope_base_and_context_come_from_the_file(self) -> None:
        config = read_model_config(FIXTURE, MODEL_ID)
        self.assertEqual(config.rope_theta, 5_000_000)
        self.assertEqual(config.context_length, 262_144)
        self.assertEqual(config.rms_norm_eps, 1e-6)

        data = json.loads(FIXTURE.read_text())
        data["rope_theta"] = 12345
        data["max_position_embeddings"] = 4096
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data))
            parsed = read_model_config(path, MODEL_ID)
        self.assertEqual(parsed.rope_theta, 12345)
        self.assertEqual(parsed.context_length, 4096)

    def test_wrong_hidden_size_is_rejected(self) -> None:
        data = json.loads(FIXTURE.read_text())
        data["hidden_size"] = 1024
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_model_config(path, MODEL_ID)

    def test_other_qwen_ids_are_rejected(self) -> None:
        for model_id in ("Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Thinking-2507"):
            with self.subTest(model_id=model_id):
                with self.assertRaises(ValueError):
                    read_model_config(FIXTURE, model_id)

    def test_checkpoint_gates_reject_a_different_architecture(self) -> None:
        replacements = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "hidden_act": "gelu",
            "attention_bias": True,
            "tie_word_embeddings": False,
            "rope_scaling": {"type": "yarn"},
            "use_sliding_window": True,
            "sliding_window": 4096,
            "torch_dtype": "float16",
        }
        for key, value in replacements.items():
            with self.subTest(field=key):
                data = json.loads(FIXTURE.read_text())
                data[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        read_model_config(path, MODEL_ID)

    def test_stop_ids_come_from_generation_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(FIXTURE.read_text())
            (root / "generation_config.json").write_text(json.dumps(GENERATION_CONFIG))
            self.assertEqual(
                read_stop_token_ids(root / "generation_config.json", root / "config.json"),
                (151645, 151643),
            )
        self.assertEqual(read_stop_token_ids(None, FIXTURE), (151645,))

    def test_one_turn_render_ends_at_the_assistant_prompt_without_think(self) -> None:
        text = render_chat(
            [{"role": "user", "content": "Hello"}],
            CHAT_TEMPLATE,
        )
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", text)
        self.assertIn("<|im_start|>user\nHello<|im_end|>\n", text)

    def test_developer_role_renders_as_system(self) -> None:
        text = render_chat(
            [
                {"role": "developer", "content": "Be brief"},
                {"role": "user", "content": "Hi"},
            ],
            CHAT_TEMPLATE,
        )
        self.assertIn("<|im_start|>system\nBe brief<|im_end|>\n", text)
        self.assertNotIn("developer", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", text)

    def test_process_exits_when_cuda_is_missing(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available")
        completed = subprocess.run(
            [sys.executable, "-m", "pylon"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("CUDA", completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
