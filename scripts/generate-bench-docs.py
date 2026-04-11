#!/usr/bin/env python3
"""Generate versioned performance docs pages from benchmark result JSONs."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import median as calc_median
from typing import Any

CATEGORY_ORDER = [
    "creation", "arithmetic", "math", "trig", "gradient", "linalg",
    "reductions", "manipulation", "io", "indexing", "bitwise",
    "sorting", "logic", "statistics", "sets", "random", "polynomials",
    "utilities", "fft",
]

DTYPE_ORDER = [
    "float64", "float32", "float16", "int64", "uint64", "int32", "uint32",
    "int16", "uint16", "int8", "uint8", "complex128", "complex64", "bool",
]

_DTYPE_RE = re.compile(
    r"\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8"
    r"|uint64|uint32|uint16|uint8|bool)$"
)


# ---------------------------------------------------------------------------
# Page definitions
# ---------------------------------------------------------------------------

PAGES: list[dict[str, Any]] = [
    {
        "id": "vs-numpy",
        "inputs": ["benchmarks/results/latest-full.json"],
        "output": "docs/{version}/performance/vs-numpy.mdx",
        "frontmatter": {
            "title": "numpy-ts vs. NumPy (Native)",
            "sidebarTitle": "vs. NumPy (Native)",
            "mode": "wide",
        },
        "component": "BenchmarkReport",
        "intro": (
            "Benchmark snapshot comparing numpy-ts against native Python NumPy "
            "(OpenBLAS-backed). "
            "Run your own via `npm run bench`."
        ),
    },
    {
        "id": "vs-pyodide",
        "inputs": ["benchmarks/results/latest-full-pyodide.json"],
        "output": "docs/{version}/performance/vs-pyodide.mdx",
        "frontmatter": {
            "title": "numpy-ts vs. NumPy (Pyodide)",
            "sidebarTitle": "vs. NumPy (Pyodide)",
            "mode": "wide",
        },
        "component": "BenchmarkReport",
        "intro": (
            "Benchmark snapshot comparing numpy-ts against "
            "[Pyodide](https://github.com/pyodide/pyodide) NumPy "
            "(WASM-compiled CPython + NumPy)."
        ),
    },
    {
        "id": "size-scaling",
        "inputs": [
            "benchmarks/results/latest-full-small.json",
            "benchmarks/results/latest-full.json",
            "benchmarks/results/latest-full-large.json",
        ],
        "output": "docs/{version}/performance/size-scaling.mdx",
        "frontmatter": {
            "title": "Performance by Array Size",
            "sidebarTitle": "Size Scaling",
            "mode": "wide",
        },
        "component": "SizeScalingReport",
        "builder": "size_scaling",
        "intro": (
            "How does numpy-ts performance scale with array size? This page shows the full picture."
        ),
    },
    {
        "id": "deno-bun",
        "inputs": ["benchmarks/results/latest-full-runtimes.json"],
        "output": "docs/{version}/performance/deno-bun.mdx",
        "frontmatter": {
            "title": "Node.js, Deno & Bun",
            "sidebarTitle": "Deno & Bun",
            "mode": "wide",
        },
        "component": "RuntimesReport",
        "builder": "runtimes",
        "intro": (
            "numpy-ts runs on Node.js, Deno, and Bun (and more). This page compares these three "
            "runtimes head-to-head against native NumPy as baseline."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_name(name: str) -> tuple[str, str | None]:
    m = _DTYPE_RE.search(name)
    return (name[: m.start()], m.group(1)) if m else (name, None)


def _benchmark_sort_key(b: dict[str, Any]) -> tuple[str, int]:
    base, dtype = _parse_name(b["name"])
    effective = dtype if dtype is not None else "float64"
    idx = DTYPE_ORDER.index(effective) if effective in DTYPE_ORDER else len(DTYPE_ORDER)
    return (base, idx)


def _geo_mean(values: list[float]) -> float:
    """Geometric mean — correct averaging method for ratios."""
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def _compute_dtype_stats(all_benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dtype_map: dict[str, list[float]] = {}
    for b in all_benchmarks:
        _, dtype = _parse_name(b["name"])
        dtype_map.setdefault(dtype or "float64", []).append(b["ratio"])
    result = []
    for dtype in DTYPE_ORDER:
        ratios = dtype_map.get(dtype)
        if not ratios:
            continue
        result.append({
            "dtype": dtype,
            "count": len(ratios),
            "avgSpeedup": round(_geo_mean(ratios), 4),
            "medianSpeedup": round(calc_median(ratios), 4),
        })
    return result


def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except Exception:
        return ts


def normalize_report(report: dict[str, Any], runtime: str = "node") -> dict[str, Any]:
    """Normalize multi-runtime format (summaries/runtimes) to single-runtime format."""
    if "summary" in report:
        return report

    normalized = {
        "timestamp": report["timestamp"],
        "environment": report["environment"],
        "summary": report["summaries"][runtime],
        "results": [],
    }
    for r in report["results"]:
        rt = r["runtimes"].get(runtime)
        if rt is None:
            continue
        normalized["results"].append({
            "name": r["name"],
            "category": r["category"],
            "numpy": r["numpy"],
            "numpyjs": rt["timing"],
            "ratio": rt["ratio"],
        })
    return normalized


# ---------------------------------------------------------------------------
# Data builder (shared across page types)
# ---------------------------------------------------------------------------

def build_benchmark_data(report: dict[str, Any], source_path: str) -> dict[str, Any]:
    """Build the data object consumed by BenchmarkReport."""
    report = normalize_report(report)
    sorted_results = sorted(report["results"], key=lambda r: r["ratio"], reverse=True)

    category_map: dict[str, list[dict[str, Any]]] = {}
    for r in sorted_results:
        speedup = 1.0 / r["ratio"] if r["ratio"] > 0 else 0.0
        row = {
            "name": r["name"],
            "ratio": round(speedup, 4),
            "numpyOps": round(r["numpy"]["ops_per_sec"], 1),
            "numpyTsOps": round(r["numpyjs"]["ops_per_sec"], 1),
        }
        category_map.setdefault(r["category"], []).append(row)

    all_benchmarks: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    for name, benchmarks in category_map.items():
        benchmarks_sorted = sorted(benchmarks, key=_benchmark_sort_key)
        avg_speedup = round(_geo_mean([b["ratio"] for b in benchmarks_sorted]), 4)
        faster_count = sum(1 for b in benchmarks_sorted if b["ratio"] >= 1)
        categories.append({
            "name": name,
            "avgSpeedup": avg_speedup,
            "count": len(benchmarks_sorted),
            "fasterCount": faster_count,
            "slowerCount": len(benchmarks_sorted) - faster_count,
            "benchmarks": benchmarks_sorted,
        })
        all_benchmarks.extend(benchmarks_sorted)

    categories.sort(key=lambda c: (
        CATEGORY_ORDER.index(c["name"]) if c["name"] in CATEGORY_ORDER else len(CATEGORY_ORDER)
    ))

    summary = report["summary"]
    geo_mean_slowdown = summary.get("avg_slowdown") or summary.get("geo_mean", 1)
    return {
        "summary": {
            "avgSpeedup": round(1.0 / geo_mean_slowdown, 4) if geo_mean_slowdown else 0,
            "medianSpeedup": round(1.0 / summary["median_slowdown"], 4) if summary["median_slowdown"] else 0,
            "bestCase": round(1.0 / summary["best_case"], 4) if summary["best_case"] else 0,
            "worstCase": round(1.0 / summary["worst_case"], 4) if summary["worst_case"] else 0,
            "totalBenchmarks": summary["total_benchmarks"],
        },
        "meta": {
            "generatedAt": format_timestamp(report["timestamp"]),
            "sourceJson": source_path,
            "runtimes": report["environment"].get("runtimes") or (
                {"node": report["environment"]["node_version"]}
                if "node_version" in report["environment"] else {}
            ),
            "pythonVersion": report["environment"].get("python_version"),
            "numpyVersion": report["environment"].get("numpy_version"),
            "numpyTsVersion": report["environment"]["numpyjs_version"],
            "machine": report["environment"].get("machine"),
        },
        "categories": categories,
        "dtypeStats": _compute_dtype_stats(all_benchmarks),
    }


# ---------------------------------------------------------------------------
# Size scaling data builder
# ---------------------------------------------------------------------------

SIZE_LABELS = ["Small (100)", "Medium (1K)", "Large (10K)"]
SIZE_KEYS = ["small", "medium", "large"]

def _normalize_bench_name(name: str) -> str:
    """Replace size-related numbers in a benchmark name for cross-size matching.

    Preserves dtype suffixes (float32, int64, etc.) and other non-size numbers.

    'add [100x100] + scalar float32' -> 'add [#] + scalar float32'
    'arange(1000)' -> 'arange(#)'
    'linspace(0, 100, 1000)' -> 'linspace(#)'
    'diag_indices n=100 int32' -> 'diag_indices n=# int32'
    """
    # First, extract and preserve the dtype suffix if present
    dtype_suffix = ""
    m = _DTYPE_RE.search(name)
    if m:
        dtype_suffix = m.group(0)  # includes leading space
        name = name[:m.start()]

    # Replace numbers inside brackets, preserving structure: [100x100] -> [#x#], [1000] -> [#]
    name = re.sub(r"\[([^\]]*)\]", lambda m: "[" + re.sub(r"\d+", "#", m.group(1)) + "]", name)
    # Replace parenthesized args: (1000), (0, 100, 1000)
    name = re.sub(r"\([^)]*\d[^)]*\)", "(#)", name)
    # Replace n=100 style params
    name = re.sub(r"(\w+=)\d+", r"\1#", name)

    return name + dtype_suffix


def build_size_scaling_data(input_paths: list[Path], repo_root: Path) -> dict[str, Any]:
    """Build the data object consumed by SizeScalingReport."""
    sizes = []
    shared_meta: dict[str, Any] | None = None

    # First pass: collect per-size category summaries and per-benchmark data keyed by normalized name
    all_size_benchmarks: list[dict[str, dict[str, Any]]] = []  # per-size: {norm_name -> {category, ratio}}

    for i, path in enumerate(input_paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        report = normalize_report(report)

        if shared_meta is None:
            shared_meta = {
                "generatedAt": format_timestamp(report["timestamp"]),
                "machine": report["environment"].get("machine"),
                "pythonVersion": report["environment"].get("python_version"),
                "numpyVersion": report["environment"].get("numpy_version"),
                "numpyTsVersion": report["environment"]["numpyjs_version"],
            }

        # Build per-category summaries
        category_map: dict[str, list[float]] = {}
        bench_map: dict[str, dict[str, Any]] = {}
        for r in report["results"]:
            speedup = 1.0 / r["ratio"] if r["ratio"] > 0 else 0.0
            ratio = round(speedup, 4)
            category_map.setdefault(r["category"], []).append(ratio)
            norm_name = _normalize_bench_name(r["name"])
            bench_map[norm_name] = {"category": r["category"], "ratio": ratio}

        all_size_benchmarks.append(bench_map)

        categories = []
        for name, ratios in category_map.items():
            avg_speedup = round(_geo_mean(ratios), 4)
            categories.append({
                "name": name,
                "avgSpeedup": avg_speedup,
                "count": len(ratios),
            })

        categories.sort(key=lambda c: (
            CATEGORY_ORDER.index(c["name"]) if c["name"] in CATEGORY_ORDER else len(CATEGORY_ORDER)
        ))

        summary = report["summary"]
        geo_mean_slowdown = summary.get("avg_slowdown") or summary.get("geo_mean", 1)
        sizes.append({
            "label": SIZE_LABELS[i],
            "summary": {
                "avgSpeedup": round(1.0 / geo_mean_slowdown, 4) if geo_mean_slowdown else 0,
                "medianSpeedup": round(1.0 / summary["median_slowdown"], 4) if summary["median_slowdown"] else 0,
                "bestCase": round(1.0 / summary["best_case"], 4) if summary["best_case"] else 0,
                "worstCase": round(1.0 / summary["worst_case"], 4) if summary["worst_case"] else 0,
                "totalBenchmarks": summary["total_benchmarks"],
            },
            "categories": categories,
        })

    # Second pass: build cross-size benchmark list grouped by category
    # Use medium (index 1) as the canonical set of benchmark names
    cross_benchmarks: dict[str, list[dict[str, Any]]] = {}
    canonical = all_size_benchmarks[1] if len(all_size_benchmarks) > 1 else all_size_benchmarks[0]
    for norm_name, info in canonical.items():
        cat = info["category"]
        bench: dict[str, Any] = {"name": norm_name}
        for si, key in enumerate(SIZE_KEYS):
            bm = all_size_benchmarks[si].get(norm_name) if si < len(all_size_benchmarks) else None
            bench[key] = round(bm["ratio"], 4) if bm else None
        cross_benchmarks.setdefault(cat, []).append(bench)

    # Sort benchmarks within each category
    for cat_benchmarks in cross_benchmarks.values():
        cat_benchmarks.sort(key=lambda b: _benchmark_sort_key(b))

    # Attach cross-size benchmarks to the output
    cross_categories = []
    for cat_name in sorted(cross_benchmarks.keys(), key=lambda n: (
        CATEGORY_ORDER.index(n) if n in CATEGORY_ORDER else len(CATEGORY_ORDER)
    )):
        cross_categories.append({
            "name": cat_name,
            "benchmarks": cross_benchmarks[cat_name],
        })

    return {"meta": shared_meta or {}, "sizes": sizes, "benchmarks": cross_categories}


# ---------------------------------------------------------------------------
# Runtimes data builder
# ---------------------------------------------------------------------------

RUNTIME_ORDER = ["node", "deno", "bun"]


def build_runtimes_data(input_path: Path, repo_root: Path) -> dict[str, Any]:
    """Build the data object consumed by RuntimesReport."""
    report = json.loads(input_path.read_text(encoding="utf-8"))
    env = report["environment"]

    meta = {
        "generatedAt": format_timestamp(report["timestamp"]),
        "machine": env.get("machine"),
        "pythonVersion": env.get("python_version"),
        "numpyVersion": env.get("numpy_version"),
        "numpyTsVersion": env["numpyjs_version"],
        "runtimes": env.get("runtimes", {}),
    }

    # Build per-runtime summaries
    summaries = {}
    for rt in RUNTIME_ORDER:
        s = report["summaries"].get(rt)
        if s is None:
            continue
        geo_mean_slowdown = s.get("avg_slowdown") or s.get("geo_mean", 1)
        summaries[rt] = {
            "avgSpeedup": round(1.0 / geo_mean_slowdown, 4) if geo_mean_slowdown else 0,
            "medianSpeedup": round(1.0 / s["median_slowdown"], 4) if s["median_slowdown"] else 0,
            "bestCase": round(1.0 / s["best_case"], 4) if s["best_case"] else 0,
            "worstCase": round(1.0 / s["worst_case"], 4) if s["worst_case"] else 0,
            "totalBenchmarks": s["total_benchmarks"],
        }

    # Build per-category cross-runtime data
    cat_map: dict[str, dict[str, Any]] = {}
    for r in report["results"]:
        cat = r["category"]
        if cat not in cat_map:
            cat_map[cat] = {"name": cat, "runtimes": {rt: {"ratios": [], "count": 0} for rt in RUNTIME_ORDER}, "benchmarks": []}

        bench: dict[str, Any] = {"name": r["name"]}
        for rt in RUNTIME_ORDER:
            rt_data = r["runtimes"].get(rt)
            if rt_data is None:
                continue
            speedup = round(1.0 / rt_data["ratio"], 4) if rt_data["ratio"] > 0 else 0.0
            ops = round(rt_data["timing"]["ops_per_sec"], 1)
            bench[rt] = {"ratio": speedup, "ops": ops}
            cat_map[cat]["runtimes"][rt]["ratios"].append(speedup)
            cat_map[cat]["runtimes"][rt]["count"] += 1

        cat_map[cat]["benchmarks"].append(bench)

    categories = []
    for cat_data in cat_map.values():
        # Sort benchmarks
        cat_data["benchmarks"] = sorted(cat_data["benchmarks"], key=lambda b: _benchmark_sort_key(b))
        # Compute per-runtime avg for this category
        for rt in RUNTIME_ORDER:
            ratios = cat_data["runtimes"][rt]["ratios"]
            cat_data["runtimes"][rt] = {
                "avgSpeedup": round(_geo_mean(ratios), 4) if ratios else 0,
                "count": len(ratios),
            }
        categories.append(cat_data)

    categories.sort(key=lambda c: (
        CATEGORY_ORDER.index(c["name"]) if c["name"] in CATEGORY_ORDER else len(CATEGORY_ORDER)
    ))

    return {
        "meta": meta,
        "runtimes": [rt for rt in RUNTIME_ORDER if rt in summaries],
        "summaries": summaries,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# MDX generation
# ---------------------------------------------------------------------------

def build_frontmatter(fm: dict[str, str]) -> str:
    lines = ["---"]
    for k, v in fm.items():
        lines.append(f'{k}: "{v}"')
    lines.append("---")
    return "\n".join(lines)


def build_page(page: dict[str, Any], repo_root: Path) -> str:
    """Generate a complete .mdx page from a page definition."""
    fm = build_frontmatter(page["frontmatter"])
    builder = page.get("builder", "default")

    if builder == "size_scaling":
        input_paths = [repo_root / p for p in page["inputs"]]
        data = build_size_scaling_data(input_paths, repo_root)
    elif builder == "runtimes":
        data = build_runtimes_data(repo_root / page["inputs"][0], repo_root)
    else:
        input_path = repo_root / page["inputs"][0]
        report = json.loads(input_path.read_text(encoding="utf-8"))
        source_display = str(input_path.relative_to(repo_root))
        data = build_benchmark_data(report, source_display)

    data_json = json.dumps(data, separators=(",", ":"))

    return f"""{fm}

import {{ {page["component"]} }} from '/snippets/{page["component"]}.jsx'

export const benchmarkData = {data_json};

{page["intro"]}

<Tip>
All benchmarks measure computation time from JS and Python, respectively. To learn more, check out [benchmark methodology](./methodology).
</Tip>

<{page["component"]} data={{benchmarkData}} />
"""


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

def detect_latest_docs_version(repo_root: Path) -> str:
    docs_config = repo_root / "docs" / "docs.json"
    if not docs_config.exists():
        return "v1.0.x"
    try:
        data = json.loads(docs_config.read_text(encoding="utf-8"))
        versions = data.get("navigation", {}).get("versions", [])
        if versions and isinstance(versions[0], dict) and isinstance(versions[0].get("version"), str):
            return versions[0]["version"]
    except Exception:
        pass
    return "v1.0.x"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    repo_root = Path(os.getcwd())
    version = detect_latest_docs_version(repo_root)

    # Single-file mode for backwards compatibility:
    #   python3 scripts/generate-bench-docs.py benchmarks/results/latest-full.json [output.mdx]
    if len(sys.argv) > 1:
        input_path = (repo_root / sys.argv[1]).resolve()
        default_output = f"docs/{version}/performance/vs-numpy.mdx"
        output_path = (repo_root / (sys.argv[2] if len(sys.argv) > 2 else default_output)).resolve()

        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1

        report = json.loads(input_path.read_text(encoding="utf-8"))
        has_summary = "summary" in report or "summaries" in report
        if not isinstance(report, dict) or "results" not in report or not has_summary:
            print("Invalid benchmark report format.", file=sys.stderr)
            return 1

        # Use vs-numpy page config as template
        page = dict(PAGES[0])
        page["inputs"] = [sys.argv[1]]
        page["output"] = str(output_path.relative_to(repo_root))
        mdx = build_page(page, repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(mdx, encoding="utf-8")
        print(f"  {output_path.relative_to(repo_root)}")
        return 0

    # Default: generate all performance pages
    print(f"Generating performance docs for {version}...")
    errors = 0
    for page in PAGES:
        output_rel = page["output"].format(version=version)
        output_path = repo_root / output_rel

        # Check all inputs exist
        missing = [p for p in page["inputs"] if not (repo_root / p).exists()]
        if missing:
            print(f"  SKIP {output_rel} (missing: {', '.join(missing)})")
            errors += 1
            continue

        mdx = build_page(page, repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(mdx, encoding="utf-8")
        print(f"  {output_rel}")

    if errors:
        print(f"\n{errors} page(s) skipped due to missing inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
