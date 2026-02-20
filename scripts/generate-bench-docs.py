#!/usr/bin/env python3
"""Generate a versioned performance docs page from benchmarks/results/latest.json."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except Exception:
        return ts


def build_doc(report: dict[str, Any], source_path: str) -> str:
    sorted_results = sorted(report["results"], key=lambda r: r["ratio"], reverse=True)

    category_map: dict[str, list[dict[str, Any]]] = {}
    for r in sorted_results:
        row = {
            "name": r["name"],
            "ratio": r["ratio"],
            "numpyMs": r["numpy"]["mean_ms"],
            "numpyTsMs": r["numpyjs"]["mean_ms"],
            "numpyOps": r["numpy"]["ops_per_sec"],
            "numpyTsOps": r["numpyjs"]["ops_per_sec"],
        }
        category_map.setdefault(r["category"], []).append(row)

    categories: list[dict[str, Any]] = []
    for name, benchmarks in category_map.items():
        avg_slowdown = sum(b["ratio"] for b in benchmarks) / len(benchmarks)
        slower_count = sum(1 for b in benchmarks if b["ratio"] >= 1)
        categories.append(
            {
                "name": name,
                "avgSlowdown": avg_slowdown,
                "count": len(benchmarks),
                "slowerCount": slower_count,
                "fasterCount": len(benchmarks) - slower_count,
                "benchmarks": benchmarks,
            }
        )

    categories.sort(key=lambda c: c["avgSlowdown"], reverse=True)

    data = {
        "summary": {
            "avgSlowdown": report["summary"]["avg_slowdown"],
            "medianSlowdown": report["summary"]["median_slowdown"],
            "bestCase": report["summary"]["best_case"],
            "worstCase": report["summary"]["worst_case"],
            "totalBenchmarks": report["summary"]["total_benchmarks"],
        },
        "meta": {
            "generatedAt": format_timestamp(report["timestamp"]),
            "sourceJson": source_path,
            "nodeVersion": report["environment"]["node_version"],
            "pythonVersion": report["environment"].get("python_version"),
            "numpyVersion": report["environment"].get("numpy_version"),
            "numpyTsVersion": report["environment"]["numpyjs_version"],
        },
        "categories": categories,
    }

    template = """---
title: Performance Benchmarks
description: Latest numpy-ts vs NumPy benchmark report from benchmarks/results/latest.json
sidebarTitle: Performance
mode: "wide"
---

import { BenchmarkReport } from '/snippets/BenchmarkReport.jsx'

export const benchmarkData = __DATA__;

# Performance Benchmarks

Latest benchmark snapshot comparing `numpy-ts` against Python NumPy.

<BenchmarkReport data={benchmarkData} />
"""

    return template.replace("__DATA__", json.dumps(data, indent=2))


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


def main() -> int:
    repo_root = Path(os.getcwd())
    input_path = (repo_root / (sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/latest.json")).resolve()
    latest_version = detect_latest_docs_version(repo_root)
    default_output = f"docs/{latest_version}/guides/performance.mdx"
    output_path = (repo_root / (sys.argv[2] if len(sys.argv) > 2 else default_output)).resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    report = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or "results" not in report or "summary" not in report or "environment" not in report:
        print("Invalid benchmark report format.", file=sys.stderr)
        return 1

    source_display = str(input_path.relative_to(repo_root))
    mdx = build_doc(report, source_display)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mdx, encoding="utf-8")
    print(f"Wrote {output_path.relative_to(repo_root)} from {source_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
