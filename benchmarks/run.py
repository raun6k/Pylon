import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
WORKLOADS = ROOT / "benchmarks" / "workloads"
PLOTS = ROOT / "benchmarks" / "plots"
VLLM_VERSION = "0.30.0"
VLLM_URL = "http://127.0.0.1:8001"
# The last flag is required on vLLM 0.30.0. Without it the response omits
# usage.prompt_tokens_details.cached_tokens, and that column cannot be scored.
VLLM_ARGV = (
    "vllm",
    "serve",
    "Qwen/Qwen3-4B-Instruct-2507",
    "--host",
    "127.0.0.1",
    "--port",
    "8001",
    "--dtype",
    "bfloat16",
    "--tensor-parallel-size",
    "1",
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.90",
    "--enable-prefix-caching",
    "--enable-per-request-metrics",
    "--enable-prompt-tokens-details",
)
LINE_ORDER = ("eager", "pylon", "vllm")

WARMUP_MESSAGES = (
    {"role": "user", "content": "Reply with the single word pylon."},
)


@dataclass(frozen=True)
class WorkloadMessage:
    role: str
    content: str


@dataclass(frozen=True)
class WorkloadRow:
    request_id: str
    prompt_tokens: int
    max_tokens: int
    temperature: float
    top_p: float
    messages: tuple[WorkloadMessage, ...]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_workload(path: Path) -> list[WorkloadRow]:
    data = json.loads(Path(path).read_text())
    rows = data["requests"] if isinstance(data, dict) else data
    loaded: list[WorkloadRow] = []
    for row in rows:
        messages = tuple(
            WorkloadMessage(role=message["role"], content=message["content"])
            for message in row["messages"]
        )
        loaded.append(
            WorkloadRow(
                request_id=row["id"],
                prompt_tokens=int(row["prompt_tokens"]),
                max_tokens=int(row["max_tokens"]),
                temperature=float(row["temperature"]),
                top_p=float(row["top_p"]),
                messages=messages,
            )
        )
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the pylon chat server.")
    parser.add_argument("--systems", default="eager")
    parser.add_argument("--workload", default="decode_heavy")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="run")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    return parser.parse_args()


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def request_json(
    base_url: str,
    path: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
    system: str = "pylon",
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{system} returned HTTP {error.code} for {path}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach {system} at {base_url}.") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{system} returned a non-JSON body for {path}.") from error


def http_ok(url: str, timeout: float) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status == 200
    except HTTPError:
        return False
    except URLError:
        return False


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GpuSampler:
    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pylon-gpu-sample", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            sample = _nvml_used_bytes(self.pid)
            if sample is not None:
                self.peak_bytes = (
                    sample if self.peak_bytes is None else max(self.peak_bytes, sample)
                )
            remaining = 1.0 - (time.perf_counter() - started)
            if self._stop.wait(max(0.0, remaining)):
                return


def parse_used_gpu_memory(text: str, pids: set[int] | None) -> int | None:
    peak = None
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            app_pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if pids is not None and app_pid not in pids:
            continue
        used = int(used_mib * 1024 * 1024)
        peak = used if peak is None else max(peak, used)
    return peak


def descendant_pids(root: int, children: dict[int, list[int]]) -> set[int]:
    found = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for child in children.get(current, ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def _process_children() -> dict[int, list[int]]:
    result = subprocess.run(
        ["ps", "-ax", "-o", "pid=", "-o", "ppid="],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or "ps failed")
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            parent = int(parts[1])
        except ValueError:
            continue
        children.setdefault(parent, []).append(pid)
    return children


def server_pids(root: int | None) -> set[int] | None:
    if root is None:
        return None
    try:
        children = _process_children()
    except (OSError, subprocess.SubprocessError):
        return {root}
    return descendant_pids(root, children)


def _nvml_used_bytes(pid: int | None) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return parse_used_gpu_memory(result.stdout, server_pids(pid))


def query_gpu_identity() -> dict[str, str | None]:
    empty = {"gpu_name": None, "driver_version": None}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return empty
    if result.returncode != 0:
        return empty
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    if not line or "," not in line:
        return empty
    name, driver = line.rsplit(",", 1)
    return {"gpu_name": name.strip() or None, "driver_version": driver.strip() or None}


def pytorch_versions() -> dict[str, str | None]:
    empty = {"pytorch_version": None, "cuda_version": None}
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.__version__); print(torch.version.cuda or '')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return empty
    if result.returncode != 0:
        return empty
    lines = [line.strip() for line in result.stdout.splitlines()]
    if not lines:
        return empty
    cuda = lines[1] if len(lines) > 1 and lines[1] else None
    return {"pytorch_version": lines[0] or None, "cuda_version": cuda}


def _published_ttft(timings: dict[str, Any]) -> float:
    return float(timings["time_to_first_token_seconds"]) - float(timings["tokenize_seconds"])


def _mean_inter_token(timings: dict[str, Any], completion_tokens: int) -> float | None:
    if completion_tokens < 2:
        return None
    gaps = timings.get("inter_token_seconds") or []
    if not gaps:
        return None
    return sum(float(gap) for gap in gaps) / len(gaps)


def run_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]] | tuple,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = request_json(
        base_url,
        "/v1/chat/completions",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {"role": message["role"], "content": message["content"]}
                if isinstance(message, dict)
                else {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    finished = time.perf_counter()
    measured = metrics_from_response(response)
    measured["response"] = response
    measured["wall_seconds"] = finished - started
    return measured


def metrics_from_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response["usage"]
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        raise RuntimeError("Response is missing cached_tokens.")
    timings = response.get("timings")
    metrics = response.get("metrics")
    completion_tokens = int(usage["completion_tokens"])
    if timings is not None:
        ttft = _published_ttft(timings)
        inter_token = _mean_inter_token(timings, completion_tokens)
        gaps = [float(gap) for gap in (timings.get("inter_token_seconds") or [])]
        accepted = timings.get("accepted_tokens_per_step")
    elif metrics is None:
        raise RuntimeError("vLLM response is missing metrics.")
    else:
        ttft = (
            float(metrics["queue_time_ms"]) + float(metrics["time_to_first_token_ms"])
        ) / 1000
        mean_itl = metrics.get("mean_itl_ms")
        inter_token = None if mean_itl is None else float(mean_itl) / 1000
        gaps = []
        timings = {}
        accepted = None
    return {
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "cached_tokens": int(cached),
        "timings": timings,
        "ttft": ttft,
        "inter_token": inter_token,
        "gaps": gaps,
        "accepted_tokens_per_step": accepted,
    }


def summarize(samples: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    ttft = [sample["ttft"] for sample in samples]
    inter = [sample["inter_token"] for sample in samples if sample["inter_token"] is not None]
    completion_tokens = sum(sample["completion_tokens"] for sample in samples)
    prefill_computed = sum(
        sample["prompt_tokens"] - sample["cached_tokens"] for sample in samples
    )
    return {
        "request_count": len(samples),
        "ttft_seconds": {"p50": percentile(ttft, 0.50), "p99": percentile(ttft, 0.99)},
        "inter_token_seconds": {
            "p50": percentile(inter, 0.50),
            "p99": percentile(inter, 0.99),
        },
        "output_tokens_per_second": (
            completion_tokens / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "prefill_tokens_computed": prefill_computed,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "mean_accepted_tokens": _mean_accepted(samples),
    }


def _mean_accepted(samples: list[dict[str, Any]]) -> float | None:
    steps = [
        int(count)
        for sample in samples
        for count in (sample.get("accepted_tokens_per_step") or ())
    ]
    if not steps:
        return None
    return sum(steps) / len(steps)


def _print_summary(system: str, workload: str, concurrency: int, summary: dict[str, Any], peak: int | None) -> None:
    ttft = summary["ttft_seconds"]
    inter = summary["inter_token_seconds"]
    peak_text = "—" if peak is None else str(peak)
    acceptance = summary.get("mean_accepted_tokens")
    acceptance_text = (
        ""
        if acceptance is None
        else f" mean_accepted_tokens={acceptance:.4f}"
    )
    print(
        f"{system} {workload} c={concurrency} "
        f"ttft_p50={ttft['p50']} ttft_p99={ttft['p99']} "
        f"inter_token_p50={inter['p50']} inter_token_p99={inter['p99']} "
        f"output_tokens_per_second={summary['output_tokens_per_second']:.4f}"
        f"{acceptance_text} "
        f"prefill_tokens_computed={summary['prefill_tokens_computed']} "
        f"peak_gpu_memory_bytes={peak_text}",
        flush=True,
    )


def _health_ok(base_url: str, timeout: float) -> dict[str, Any] | None:
    try:
        return request_json(base_url, "/health", timeout=min(timeout, 5))
    except RuntimeError:
        return None


def _wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            return request_json(base_url, "/health", timeout=5)
        except RuntimeError as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"pylon /health was not ready at {base_url}.") from last_error


def _wait_vllm(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    health = f"{base_url.rstrip('/')}/health"
    while time.perf_counter() < deadline:
        if http_ok(health, timeout=5):
            version_body = request_json(
                base_url, "/version", timeout=timeout, system="vllm"
            )
            version = str(version_body.get("version") or "")
            if version != VLLM_VERSION:
                raise RuntimeError(
                    f"vLLM reported version {version or 'unknown'}; "
                    f"{VLLM_VERSION} is required."
                )
            models = request_json(base_url, "/v1/models", timeout=timeout, system="vllm")
            data = models.get("data") or []
            if not data or "id" not in data[0]:
                raise RuntimeError("vLLM /v1/models did not report a model id.")
            return {"model": str(data[0]["id"]), "vllm_version": version}
        time.sleep(1)
    raise RuntimeError(f"vLLM /health was not ready at {base_url}.")


def system_environ(base: dict[str, str], system: str) -> dict[str, str]:
    env = dict(base)
    if system == "eager":
        env["PYLON_CUDA_GRAPHS"] = "false"
        env["PYLON_PREFIX_CACHE"] = "false"
        env["PYLON_SPECULATE_K"] = "1"
        env["PYLON_ADMIT_SKIP"] = "0"
    elif system == "pylon":
        env["PYLON_CUDA_GRAPHS"] = "true"
        env["PYLON_PREFIX_CACHE"] = "false"
        env["PYLON_SPECULATE_K"] = "1"
        env["PYLON_ADMIT_SKIP"] = "0"
    return env


def result_flags(
    system: str, env: dict[str, str] | None = None
) -> dict[str, object]:
    speculate_k = 1
    if env is not None and "PYLON_SPECULATE_K" in env:
        speculate_k = int(env["PYLON_SPECULATE_K"])
    if system == "pylon":
        admit_skip = 0
        if env is not None and "PYLON_ADMIT_SKIP" in env:
            admit_skip = int(env["PYLON_ADMIT_SKIP"])
    else:
        admit_skip = 0
    return {
        "cuda_graphs": system == "pylon",
        "speculation": speculate_k > 1,
        "prefix_cache": system not in {"eager", "pylon"},
        "admit_skip": admit_skip,
    }


def reuse_running_server(systems: list[str], system: str) -> bool:
    return len(systems) == 1 and system != "vllm"


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _spawn_system(system: str) -> tuple[subprocess.Popen[bytes], str]:
    if system == "vllm":
        command = list(VLLM_ARGV)
        env = os.environ.copy()
        url = VLLM_URL
    else:
        env = system_environ(os.environ.copy(), system)
        command = [sys.executable, "-m", "pylon"]
        url = "http://127.0.0.1:8000"
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError as error:
        if system == "vllm":
            raise RuntimeError(
                f"vLLM {VLLM_VERSION} was not found on PATH."
            ) from error
        raise RuntimeError("Cannot start pylon.") from error
    return process, url


def hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def cached_snapshot_revision() -> str | None:
    ref = hub_dir() / "models--Qwen--Qwen3-4B-Instruct-2507" / "refs" / "main"
    if not ref.is_file():
        return None
    revision = ref.read_text().strip()
    return revision or None


def verify_prompt_tokens(revision: str | None, rows: list[WorkloadRow]) -> None:
    if not revision:
        raise RuntimeError("Cannot tokenize workload rows without a snapshot revision.")
    snapshot = hub_dir() / "models--Qwen--Qwen3-4B-Instruct-2507" / "snapshots" / revision
    tokenizer_path = snapshot / "tokenizer.json"
    template_path = snapshot / "tokenizer_config.json"
    if not tokenizer_path.is_file() or not template_path.is_file():
        raise RuntimeError(f"Cannot tokenize workload rows from {snapshot}.")
    from tokenizers import Tokenizer
    from jinja2.sandbox import SandboxedEnvironment

    client = Tokenizer.from_file(str(tokenizer_path))
    document = json.loads(tokenizer_path.read_text())
    specials = {}
    for item in document.get("added_tokens", []):
        content = item.get("content")
        token_id = client.token_to_id(content) if content else None
        if content and token_id is not None:
            specials[content] = token_id
    ordered = sorted(specials, key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(token) for token in ordered) + ")")
    template_text = json.loads(template_path.read_text())["chat_template"]
    environment = SandboxedEnvironment()
    environment.filters["tojson"] = lambda value: json.dumps(value)
    template = environment.from_string(template_text)
    for row in rows:
        messages = [
            {
                "role": "system" if message.role == "developer" else message.role,
                "content": message.content,
            }
            for message in row.messages
        ]
        rendered = template.render(messages=messages, tools=None, add_generation_prompt=True)
        token_ids: list[int] = []
        for part in filter(None, pattern.split(rendered)):
            if part in specials:
                token_ids.append(specials[part])
            else:
                token_ids.extend(client.encode(part).ids)
        if len(token_ids) != row.prompt_tokens:
            raise SystemExit(
                f"{row.request_id} tokenized to {len(token_ids)}, expected {row.prompt_tokens}."
            )


def _run_rows(
    base_url: str,
    model: str,
    rows: list[WorkloadRow],
    *,
    concurrency: int,
    timeout: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any] | None] = [None] * len(rows)

    def one(index: int, row: WorkloadRow) -> tuple[int, dict[str, Any]]:
        sample = run_completion(
            base_url,
            model,
            row.messages,
            max_tokens=row.max_tokens,
            temperature=row.temperature,
            top_p=row.top_p,
            timeout=timeout,
        )
        sample["id"] = row.request_id
        return index, sample

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(rows)))) as pool:
        futures = [pool.submit(one, index, row) for index, row in enumerate(rows)]
        for future in as_completed(futures):
            index, sample = future.result()
            samples[index] = sample
    return [sample for sample in samples if sample is not None]


def decode_heavy_series(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    ttft: dict[str, list[tuple[int, float]]] = {}
    throughput: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        if record.get("workload") != "decode_heavy":
            continue
        system = str(record["system"])
        concurrency = int(record["concurrency"])
        summary = record["summary"]
        p50 = summary["ttft_seconds"]["p50"]
        if p50 is not None:
            ttft.setdefault(system, []).append((concurrency, float(p50)))
        rate = summary.get("output_tokens_per_second")
        if rate is not None:
            throughput.setdefault(system, []).append((concurrency, float(rate)))
    return {"ttft": ttft, "throughput": throughput}


def ordered_lines(points_by_system: dict[str, list[tuple[int, float]]]) -> list[str]:
    return [name for name in LINE_ORDER if points_by_system.get(name)]


def should_write_plots(args: argparse.Namespace, records: list[dict[str, Any]]) -> bool:
    if args.limit or args.repeats < 3:
        return False
    seen: dict[str, set[int]] = {}
    for record in records:
        if record.get("workload") != "decode_heavy":
            continue
        seen.setdefault(str(record["system"]), set()).add(int(record["concurrency"]))
    return all({1, 8, 16} <= seen.get(system, set()) for system in LINE_ORDER)


def draw_comparison(
    points_by_system: dict[str, list[tuple[int, float]]],
    ylabel: str,
    axis: Any,
) -> None:
    for system in ordered_lines(points_by_system):
        points = sorted(points_by_system[system])
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=system,
        )
    axis.set_xlabel("Concurrency")
    axis.set_ylabel(ylabel)
    ticks = sorted({point[0] for points in points_by_system.values() for point in points})
    if ticks:
        axis.set_xticks(ticks)
    if ordered_lines(points_by_system):
        axis.legend()


def write_comparison_plots(
    records: list[dict[str, Any]], directory: Path
) -> list[Path]:
    series = decode_heavy_series(records)
    if not series["ttft"] and not series["throughput"]:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    specs = (
        (
            "ttft_vs_concurrency.png",
            series["ttft"],
            "Time to first token p50 (seconds)",
        ),
        (
            "output_tokens_per_second_vs_concurrency.png",
            series["throughput"],
            "Output tokens per second",
        ),
    )
    written: list[Path] = []
    for filename, points, ylabel in specs:
        figure, axis = plt.subplots()
        try:
            draw_comparison(points, ylabel, axis)
            path = directory / filename
            figure.savefig(path)
        finally:
            plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    systems = _split_list(args.systems)
    workloads = _split_list(args.workload)
    concurrencies = [int(item) for item in _split_list(args.concurrency)]
    identity = query_gpu_identity()
    runtime = pytorch_versions()
    records = []
    for system in systems:
        base_url = VLLM_URL if system == "vllm" else args.base_url
        existing = (
            _health_ok(base_url, args.timeout)
            if reuse_running_server(systems, system)
            else None
        )
        process = None
        try:
            if existing is None:
                process, base_url = _spawn_system(system)
                health = (
                    _wait_vllm(base_url, args.timeout)
                    if system == "vllm"
                    else _wait_health(base_url, args.timeout)
                )
            else:
                health = existing
            spawn_env = (
                None if system == "vllm" else system_environ(os.environ.copy(), system)
            )
            scheduler = health.get("scheduler") or {}
            if system == "vllm":
                batch_cap = None
                prefill_chunk = None
            else:
                batch_cap = scheduler.get("max_batch_size")
                if batch_cap is None:
                    batch_cap = int((spawn_env or {}).get("PYLON_MAX_BATCH_SIZE", "8"))
                prefill_chunk = int(
                    (spawn_env or {}).get("PYLON_PREFILL_CHUNK_SIZE", "256")
                )
            model = health["model"]
            for workload_name in workloads:
                path = WORKLOADS / f"{workload_name}.json"
                rows = load_workload(path)
                revision = health.get("model_revision") or cached_snapshot_revision()
                verify_prompt_tokens(revision, rows)
                if args.limit:
                    rows = rows[: args.limit]
                for concurrency in concurrencies:
                    run_completion(
                        base_url,
                        model,
                        WARMUP_MESSAGES,
                        max_tokens=16,
                        temperature=0,
                        top_p=1,
                        timeout=args.timeout,
                    )
                    if system != "vllm":
                        request_json(
                            base_url,
                            "/internal/profile/reset",
                            timeout=args.timeout,
                            payload={},
                            system=system,
                        )
                    sampler = GpuSampler(process.pid if process is not None else None)
                    sampler.start()
                    started = time.perf_counter()
                    samples: list[dict[str, Any]] = []
                    try:
                        for repeat in range(args.repeats):
                            if repeat:
                                run_completion(
                                    base_url,
                                    model,
                                    WARMUP_MESSAGES,
                                    max_tokens=16,
                                    temperature=0,
                                    top_p=1,
                                    timeout=args.timeout,
                                )
                            samples.extend(
                                _run_rows(
                                    base_url,
                                    model,
                                    rows,
                                    concurrency=concurrency,
                                    timeout=args.timeout,
                                )
                            )
                    finally:
                        elapsed = time.perf_counter() - started
                        sampler.stop()
                    summary = summarize(samples, elapsed)
                    peak_reserved = None
                    if system != "vllm":
                        after = request_json(
                            base_url, "/health", timeout=args.timeout, system=system
                        )
                        peak_reserved = after.get("peak_reserved_bytes")
                    summary["peak_reserved_bytes"] = peak_reserved
                    summary["peak_gpu_memory_bytes"] = sampler.peak_bytes
                    _print_summary(
                        system, workload_name, concurrency, summary, sampler.peak_bytes
                    )
                    records.append(
                        {
                            "schema_version": 1,
                            "label": args.label,
                            "system": system,
                            "workload": workload_name,
                            "concurrency": concurrency,
                            "repeats": args.repeats,
                            "limit": args.limit,
                            "git_revision": git_revision(),
                            "gpu_name": (health.get("memory") or {}).get("gpu")
                            or identity["gpu_name"],
                            "model_id": model,
                            "snapshot_revision": revision,
                            "dataset_sha256": dataset_sha256(path),
                            "batch_cap": batch_cap,
                            "prefill_chunk_size": prefill_chunk,
                            "vllm_version": health.get("vllm_version"),
                            "vllm_argv": list(VLLM_ARGV) if system == "vllm" else None,
                            "summary": summary,
                            "flags": result_flags(system, spawn_env),
                            "samples": samples,
                        }
                    )
        finally:
            if process is not None:
                stop_process(process)
    if should_write_plots(args, records):
        for plot in write_comparison_plots(records, PLOTS):
            print(f"wrote {plot}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in args.label)
    path = RESULTS / f"{stamp}-{safe}.json"
    observed_vllm = next(
        (record.get("vllm_version") for record in records if record["system"] == "vllm"),
        None,
    )
    record = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": sys.argv,
        "pytorch_version": runtime["pytorch_version"],
        "cuda_version": runtime["cuda_version"],
        "driver_version": identity["driver_version"],
        "gpu_name": identity["gpu_name"],
        "vllm_version": observed_vllm,
        "vllm_argv": list(VLLM_ARGV) if "vllm" in systems else None,
        "runs": records,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
