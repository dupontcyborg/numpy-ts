#!/usr/bin/env python3
"""
Audit documentation API coverage against implemented numpy-ts exports.

This script:
1. Runs scripts/audit-numpyts-api.ts to generate fresh function export data.
2. Detects the latest docs version folder (e.g. docs/v1.0.x/api).
3. Extracts function signatures from docs API pages.
4. Reports:
   - Implemented functions missing from docs
   - Doc signatures that do not map to implemented functions

Usage:
    python scripts/compare-docs-api-coverage.py
    python scripts/compare-docs-api-coverage.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DOCS_ROOT = REPO_ROOT / 'docs'


# Functions exported but intentionally not represented as API doc signatures.
IGNORED_EXPORTS = {
    'Complex',
    'NDArray',
    'NDArrayCore',
    # Namespaces/classes
    'linalg',
    'random',
    'random.Generator',
    'fft',
    # Types / errors / constants
    'UnsupportedDTypeError',
    'InvalidNpyError',
    'SUPPORTED_DTYPES',
    'DTYPE_TO_DESCR',
}

# Aliases that may not have dedicated signature blocks but should be mentioned in docs.
ALIAS_EXPORTS = {
    'acos',
    'acosh',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'bitwise_invert',
    'bitwise_left_shift',
    'bitwise_right_shift',
    'concat',
    'conjugate',
    'cumulative_prod',
    'cumulative_sum',
    'delete',
    'max',
    'min',
    'round_',
    'row_stack',
    'var',
}


def run_numpyts_audit() -> bool:
    """Generate scripts/numpyts-api-audit.json from the canonical audit script."""
    print("1. Auditing numpy-ts exports...")
    result = subprocess.run(
        ['npx', 'tsx', str(SCRIPT_DIR / 'audit-numpyts-api.ts')],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("   ⚠️  Could not refresh export audit via tsx.")
        print("   Falling back to existing scripts/numpyts-api-audit.json")
        return False
    print("   ✅ Export audit complete\n")
    return True


def load_numpyts_functions() -> set[str]:
    """Load flattened function names from scripts/numpyts-api-audit.json."""
    audit_file = SCRIPT_DIR / 'numpyts-api-audit.json'
    if not audit_file.exists():
        print(f"Missing audit file: {audit_file}")
        sys.exit(1)

    with open(audit_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_funcs = set(data.get('all_functions', {}).keys())

    filtered: set[str] = set()
    for func in all_funcs:
        if func in IGNORED_EXPORTS:
            continue
        # Ignore uppercase-named exported callables/classes
        last = func.split('.')[-1]
        if last and last[0].isupper():
            continue
        filtered.add(func)

    return filtered


def detect_latest_api_dir() -> Path:
    """
    Detect latest docs API directory.

    Preference:
    1) docs/docs.json redirect "/api" -> "/<version>/api/overview"
    2) Highest lexicographic docs/v*/api folder.
    """
    docs_json = DOCS_ROOT / 'docs.json'
    if docs_json.exists():
        try:
            payload = json.loads(docs_json.read_text(encoding='utf-8'))
            redirects = payload.get('redirects', [])
            for item in redirects:
                if item.get('source') == '/api':
                    dest = item.get('destination', '')
                    m = re.match(r'^/([^/]+)/api/overview$', dest)
                    if m:
                        candidate = DOCS_ROOT / m.group(1) / 'api'
                        if candidate.exists():
                            return candidate
        except Exception:
            # Fallback to directory scan below
            pass

    candidates = sorted(
        p for p in DOCS_ROOT.glob('v*/api') if p.is_dir()
    )
    if not candidates:
        print("Could not find docs version API directory under docs/v*/api")
        sys.exit(1)
    return candidates[-1]


def extract_doc_functions(api_dir: Path) -> set[str]:
    """Extract declared function names from API MDX pages."""
    fn_re = re.compile(r'^\s*function\s+([A-Za-z_][A-Za-z0-9_\.]*)\s*\(')
    functions: set[str] = set()

    for path in api_dir.rglob('*.mdx'):
        rel = path.relative_to(api_dir).as_posix()
        text = path.read_text(encoding='utf-8')

        for line in text.splitlines():
            m = fn_re.match(line)
            if not m:
                continue
            name = m.group(1)
            if name == 'functionName':  # placeholder in overview docs
                continue
            functions.add(normalize_doc_name(name, rel))

    return functions


def extract_docs_text(api_dir: Path) -> str:
    """Concatenate API docs text for lightweight mention checks."""
    parts: list[str] = []
    for path in api_dir.rglob('*.mdx'):
        parts.append(path.read_text(encoding='utf-8'))
    return "\n".join(parts)


def is_mentioned_in_docs(name: str, docs_text: str) -> bool:
    """True if a name appears as a standalone token in docs text."""
    return re.search(rf'(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])', docs_text) is not None


def normalize_doc_name(name: str, rel_path: str) -> str:
    """
    Normalize doc signature names to exported function naming.

    - linalg pages contain top-level signatures for many functions;
      do not auto-prefix by folder.
    - random API pages generally declare bare names -> random.<name>
    - fft API pages declare bare names -> fft.<name>
    """
    if '.' in name:
        return name

    if rel_path.startswith('random/'):
        return f'random.{name}'
    if rel_path.startswith('fft/'):
        return f'fft.{name}'
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare docs API coverage to implemented exports.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show full missing/extra lists')
    parser.add_argument(
        '--no-refresh',
        action='store_true',
        help='Skip running audit-numpyts-api.ts and use existing JSON',
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DOCS API COVERAGE AUDIT")
    print("=" * 70)

    if not args.no_refresh:
        run_numpyts_audit()
    else:
        print("1. Skipping export refresh (--no-refresh)\n")

    implemented = load_numpyts_functions()
    api_dir = detect_latest_api_dir()
    documented = extract_doc_functions(api_dir)
    docs_text = extract_docs_text(api_dir)

    alias_mentioned = sorted(
        a for a in (implemented & ALIAS_EXPORTS)
        if is_mentioned_in_docs(a, docs_text)
    )
    linalg_namespace_mentioned = sorted(
        f for f in implemented
        if f.startswith('linalg.') and is_mentioned_in_docs(f, docs_text)
    )
    effectively_documented = documented | set(alias_mentioned) | set(linalg_namespace_mentioned)

    missing_from_docs = sorted(implemented - effectively_documented)
    docs_without_impl = sorted(documented - implemented)

    coverage = (100.0 * (len(implemented) - len(missing_from_docs)) / len(implemented)) if implemented else 0.0
    report = {
        'latest_api_dir': str(api_dir.relative_to(REPO_ROOT)),
        'implemented_count': len(implemented),
        'documented_count': len(documented),
        'documented_effective_count': len(effectively_documented),
        'documented_implemented_count': len(implemented) - len(missing_from_docs),
        'coverage_percent': coverage,
        'alias_mentioned_in_docs': alias_mentioned,
        'linalg_namespace_mentions': linalg_namespace_mentioned,
        'missing_from_docs': missing_from_docs,
        'docs_without_implementation_match': docs_without_impl,
    }
    output_file = SCRIPT_DIR / 'docs-api-audit.json'
    try:
        output_file.write_text(json.dumps(report, indent=2) + "\n", encoding='utf-8')
    except PermissionError:
        # Sandbox fallback: still produce a report in a writable location.
        output_file = DOCS_ROOT / 'docs-api-audit.json'
        output_file.write_text(json.dumps(report, indent=2) + "\n", encoding='utf-8')

    print(f"2. Latest docs API folder: {api_dir.relative_to(REPO_ROOT)}")
    print(f"3. Implemented functions (filtered): {len(implemented)}")
    print(f"4. Documented function signatures:   {len(documented)}")
    print(f"5. Aliases mentioned in docs:       {len(alias_mentioned)}")
    print(f"6. linalg.* mentions in docs:       {len(linalg_namespace_mentioned)}")
    print()
    print(f"Coverage (implemented documented): {len(implemented) - len(missing_from_docs)}/{len(implemented)} ({coverage:.1f}%)")
    print(f"Missing from docs: {len(missing_from_docs)}")
    print(f"Docs without implementation match: {len(docs_without_impl)}")
    try:
        report_rel = output_file.relative_to(REPO_ROOT)
    except ValueError:
        report_rel = output_file
    print(f"Report written: {report_rel}")

    if args.verbose:
        if missing_from_docs:
            print("\nMissing from docs:")
            for i, name in enumerate(missing_from_docs, 1):
                print(f"  {i:3d}. {name}")
        if docs_without_impl:
            print("\nDocs without implementation match:")
            for i, name in enumerate(docs_without_impl, 1):
                print(f"  {i:3d}. {name}")
    else:
        if missing_from_docs:
            print("\nTop missing from docs:")
            for name in missing_from_docs[:25]:
                print(f"  - {name}")
            if len(missing_from_docs) > 25:
                print(f"  ... and {len(missing_from_docs) - 25} more")

    # Exit non-zero if there are undocumented implemented functions.
    sys.exit(1 if missing_from_docs else 0)


if __name__ == '__main__':
    main()
