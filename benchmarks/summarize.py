"""Turn benchmark result files into the markdown tables and plots the README shows.

    python benchmarks/summarize.py pylon=<file> vllm=<file> before=<file> eager=<file>

Each argument names one column and points at a result JSON written by benchmarks/run.py.
A column can come from several files: repeat the name and the rows are merged.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "benchmarks" / "tables"
PLOTS = ROOT / "benchmarks" / "plots"
WORKLOADS = ("decode_heavy", "prefill_heavy", "shared_prefix")
CONCURRENCY = (1, 8, 16)
COLUMN_ORDER = ("eager", "before", "pylon", "vllm")
COLUMN_LABEL = {
    "eager": "eager",
    "before": "pylon (before)",
    "pylon": "pylon",
    "vllm": "vLLM 0.30.0",
}
COLUMN_STYLE = {
    "eager": ("#888888", "s", ":"),
    "before": ("#d95f02", "v", "--"),
    "pylon": ("#1b9e77", "o", "-"),
    "vllm": ("#7570b3", "^", "-"),
}


def load(arguments: list[str]) -> tuple[dict, dict]:
    rows: dict[tuple[str, str, int], dict] = {}
    meta: dict[str, dict] = {}
    for argument in arguments:
        name, _, path = argument.partition("=")
        record = json.loads(Path(path).read_text())
        for run in record["runs"]:
            rows[(name, run["workload"], int(run["concurrency"]))] = run
        info = meta.setdefault(name, {})
        files = info.setdefault("files", {})
        for run in record["runs"]:
            files[run["workload"]] = Path(path).name
        info.update(
            gpu=record["gpu_name"],
            driver=record["driver_version"],
            torch=record["pytorch_version"],
            cuda=record["cuda_version"],
        )
        first = record["runs"][0]
        info.update(
            batch_cap=first["batch_cap"],
            chunk=first["prefill_chunk_size"],
            snapshot=first["snapshot_revision"],
            vllm=record.get("vllm_version"),
        )
    return rows, meta


def ms(value: float | None) -> str:
    return "–" if value is None else f"{value * 1000:,.0f}"


def cell(run: dict | None, field: str) -> str:
    if run is None:
        return "–"
    summary = run["summary"]
    if field == "tok":
        return f"{summary['output_tokens_per_second']:.1f}"
    if field == "ttft":
        return f"{ms(summary['ttft_seconds']['p50'])} / {ms(summary['ttft_seconds']['p99'])}"
    if field == "itl":
        return f"{ms(summary['inter_token_seconds']['p50'])} / {ms(summary['inter_token_seconds']['p99'])}"
    if field == "prefill":
        return f"{summary['prefill_tokens_computed']:,}"
    if field == "mem":
        peak = summary.get("peak_gpu_memory_bytes")
        return "–" if peak is None else f"{peak / 2**30:.1f}"
    raise ValueError(field)


FIELDS = (
    ("tok", "Output tokens/s"),
    ("ttft", "TTFT p50 / p99 (ms)"),
    ("itl", "Inter-token p50 / p99 (ms)"),
    ("prefill", "Prefill tokens computed"),
)


def workload_tables(rows: dict, columns: list[str]) -> str:
    out = []
    for workload in WORKLOADS:
        out.append(f"### {workload}\n")
        header = ["Concurrency", "Metric", *(COLUMN_LABEL[name] for name in columns)]
        out.append("| " + " | ".join(header) + " |")
        out.append("| --- | --- | " + " | ".join("---:" for _ in columns) + " |")
        for concurrency in CONCURRENCY:
            for field, title in FIELDS:
                cells = [cell(rows.get((name, workload, concurrency)), field) for name in columns]
                if all(value == "–" for value in cells):
                    continue
                lead = str(concurrency) if field == "tok" else ""
                out.append(f"| {lead} | {title} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def ratio_table(rows: dict, numerator: str, denominator: str, title: str) -> str:
    out = [f"### {title}\n", "| Workload | Concurrency | Output tokens/s | TTFT p50 | Inter-token p50 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for workload in WORKLOADS:
        for concurrency in CONCURRENCY:
            top = rows.get((numerator, workload, concurrency))
            bottom = rows.get((denominator, workload, concurrency))
            if top is None or bottom is None:
                continue
            a, b = top["summary"], bottom["summary"]
            tok = a["output_tokens_per_second"] / b["output_tokens_per_second"]
            ttft = a["ttft_seconds"]["p50"] / b["ttft_seconds"]["p50"]
            itl = a["inter_token_seconds"]["p50"] / b["inter_token_seconds"]["p50"]
            out.append(f"| {workload} | {concurrency} | {tok:.2f}x | {ttft:.2f}x | {itl:.2f}x |")
    out.append("")
    return "\n".join(out)


def setup_lines(meta: dict, columns: list[str]) -> str:
    any_meta = next(iter(meta.values()))
    lines = [
        f"GPU {any_meta['gpu']}, driver {any_meta['driver']}, PyTorch {any_meta['torch']}, "
        f"CUDA {any_meta['cuda']}, snapshot `{any_meta['snapshot']}`.",
    ]
    for name in columns:
        info = meta.get(name)
        if info is None or name == "vllm":
            continue
        lines.append(
            f"- {COLUMN_LABEL[name]}: batch cap {info['batch_cap']}, prefill chunk {info['chunk']}."
        )
    lines.append("\nRows come from these files in `benchmarks/results/`:\n")
    for name in columns:
        by_file: dict[str, list[str]] = {}
        for workload in WORKLOADS:
            file = meta[name]["files"].get(workload)
            if file:
                by_file.setdefault(file, []).append(workload)
        parts = "; ".join(f"{', '.join(w)} from `{f}`" for f, w in by_file.items())
        lines.append(f"- {COLUMN_LABEL[name]}: {parts}")
    return "\n".join(lines)


def headline_tables(rows: dict, columns: list[str]) -> str:
    """Three compact tables for the README: one metric each, a row per workload and concurrency."""
    metrics = (
        ("Output tokens per second", lambda r: f"{r['summary']['output_tokens_per_second']:.1f}"),
        ("Time to first token p50 (ms)", lambda r: ms(r["summary"]["ttft_seconds"]["p50"])),
        ("Inter-token p50 (ms)", lambda r: ms(r["summary"]["inter_token_seconds"]["p50"])),
    )
    out = []
    for title, render in metrics:
        out.append(f"**{title}**\n")
        out.append("| Workload | Concurrency | " + " | ".join(COLUMN_LABEL[c] for c in columns) + " |")
        out.append("| --- | ---: | " + " | ".join("---:" for _ in columns) + " |")
        for workload in WORKLOADS:
            for concurrency in CONCURRENCY:
                runs = [rows.get((name, workload, concurrency)) for name in columns]
                if all(run is None for run in runs):
                    continue
                cells = ["–" if run is None else render(run) for run in runs]
                out.append(f"| {workload} | {concurrency} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def plots(rows: dict, columns: list[str]) -> list[Path]:
    """The two figures from the benchmark spec, from decode_heavy.

    Concurrency 1 to 16 spans a 15x range, so each concurrency gets its own panel and its
    own axis. Bars start at zero, carry their value, and show each system's value as a
    percentage of vLLM's.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    specs = (
        (
            "ttft_vs_concurrency.png",
            "Time to first token p50 (seconds)",
            lambda s: s["ttft_seconds"]["p50"],
            "{:.3f}",
        ),
        (
            "output_tokens_per_second_vs_concurrency.png",
            "Output tokens per second",
            lambda s: s["output_tokens_per_second"],
            "{:.1f}",
        ),
    )
    systems = [name for name in columns if name != "before"]
    PLOTS.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, ylabel, value, number in specs:
        figure, axes = plt.subplots(1, len(CONCURRENCY), figsize=(9.0, 3.8), dpi=140)
        for axis, concurrency in zip(axes, CONCURRENCY):
            values = {
                name: value(rows[(name, "decode_heavy", concurrency)]["summary"])
                for name in systems
                if (name, "decode_heavy", concurrency) in rows
            }
            reference = values.get("vllm")
            top = max(values.values())
            for position, name in enumerate(systems):
                if name not in values:
                    continue
                color = COLUMN_STYLE[name][0]
                axis.bar(position, values[name], color=color, width=0.7, label=COLUMN_LABEL[name])
                label = number.format(values[name])
                if reference and name != "vllm":
                    label += f"\n({100 * values[name] / reference:.0f}%)"
                axis.text(
                    position, values[name] + top * 0.015, label, ha="center", va="bottom", fontsize=8
                )
            axis.set_title(f"Concurrency {concurrency}", fontsize=10)
            axis.set_xticks([])
            axis.set_ylim(0, top * 1.22)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            for edge in ("top", "right"):
                axis.spines[edge].set_visible(False)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
        figure.suptitle("decode_heavy  (percentages are relative to vLLM)", fontsize=10)
        figure.tight_layout(rect=(0, 0.07, 1, 0.95))
        path = PLOTS / filename
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows, meta = load(sys.argv[1:])
    columns = [name for name in COLUMN_ORDER if name in meta]
    TABLES.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Benchmark summary\n",
            setup_lines(meta, columns),
            "\nTTFT is server time from the request joining the queue to its first token. "
            "32 requests, 3 repeats, temperature 0. pylon and vLLM run with the prefix cache on for "
            "shared_prefix only; eager never has it.\n",
            workload_tables(rows, columns),
        ]
        + (
            [ratio_table(rows, "pylon", "before", "pylon vs pylon (before)")]
            if "before" in columns
            else []
        )
        + (
            [ratio_table(rows, "pylon", "vllm", "pylon vs vLLM 0.30.0")]
            if "vllm" in columns
            else []
        )
    )
    path = TABLES / "summary.md"
    path.write_text(text)
    print(f"wrote {path}")
    headline = TABLES / "headline.md"
    headline.write_text(headline_tables(rows, columns))
    print(f"wrote {headline}")
    for plot in plots(rows, columns):
        print(f"wrote {plot}")


if __name__ == "__main__":
    main()
