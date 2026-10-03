import json
import re
from pathlib import Path

from jinja2.sandbox import SandboxedEnvironment
from tokenizers import Tokenizer


def render_chat(messages: list[dict], chat_template: str) -> str:
    rendered = []
    for message in messages:
        role = "system" if message["role"] == "developer" else message["role"]
        rendered.append({"role": role, "content": message["content"]})
    environment = SandboxedEnvironment()
    environment.filters["tojson"] = lambda value: json.dumps(value)
    template = environment.from_string(chat_template)
    return template.render(
        messages=rendered,
        tools=None,
        add_generation_prompt=True,
    )


def read_stop_token_ids(
    generation_config_path: Path | None,
    config_path: Path,
) -> tuple[int, ...]:
    if generation_config_path is not None and Path(generation_config_path).is_file():
        data = json.loads(Path(generation_config_path).read_text())
        eos = data["eos_token_id"]
        if isinstance(eos, int):
            return (eos,)
        return tuple(int(token_id) for token_id in eos)
    config = json.loads(Path(config_path).read_text())
    eos = config["eos_token_id"]
    if isinstance(eos, int):
        return (eos,)
    return tuple(int(token_id) for token_id in eos)


class CheckpointTokenizer:
    def __init__(self, snapshot: Path, stop_token_ids: tuple[int, ...]) -> None:
        snapshot = Path(snapshot)
        self.client = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        document = json.loads((snapshot / "tokenizer.json").read_text())
        specials = [item["content"] for item in document.get("added_tokens", [])]
        self.special_to_id = {
            token: token_id
            for token in specials
            if (token_id := self.client.token_to_id(token)) is not None
        }
        ordered = sorted(self.special_to_id, key=len, reverse=True)
        self._pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in ordered) + ")"
        )
        template = json.loads((snapshot / "tokenizer_config.json").read_text())
        self.chat_template = template["chat_template"]
        self.stop_token_ids = stop_token_ids

    def encode_chat(self, messages: list[tuple[str, str]]) -> list[int]:
        rendered = render_chat(
            [{"role": role, "content": content} for role, content in messages],
            self.chat_template,
        )
        token_ids: list[int] = []
        for part in filter(None, self._pattern.split(rendered)):
            if part in self.special_to_id:
                token_ids.append(self.special_to_id[part])
            else:
                token_ids.extend(self.client.encode(part).ids)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.client.decode(token_ids, skip_special_tokens=True)
