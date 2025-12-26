#!/usr/bin/env python3
"""
Compare numpy-ts vs NumPy API and update documentation.

Compares:
- Top-level functions vs top-level functions (ignoring if also methods)
- ndarray methods vs NDArray methods (ignoring if also functions)

Usage:
    python scripts/compare-api-coverage.py              # Update README
    python scripts/compare-api-coverage.py --verbose    # Show detailed gaps
    python scripts/compare-api-coverage.py -v           # Short form
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_audits():
    """Run both audit scripts to generate fresh JSON files."""
    scripts_dir = Path(__file__).parent

    print("Running audits to generate fresh data...\n")

    # Run NumPy audit
    print("1. Auditing NumPy API...")
    result = subprocess.run(
        [sys.executable, scripts_dir / 'audit-numpy-api.py'],
        cwd=scripts_dir.parent,
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        print(f"Error running NumPy audit:\n{result.stderr}")
        sys.exit(1)

    # Run numpy-ts audit
    print("2. Auditing numpy-ts API...")
    result = subprocess.run(
        ['npx', 'tsx', scripts_dir / 'audit-numpyts-api.ts'],
        cwd=scripts_dir.parent,
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        print(f"Error running numpy-ts audit:\n{result.stderr}")
        sys.exit(1)

    print("✅ Audits complete\n")


def load_json(filename):
    """Load JSON file."""
    filepath = Path(__file__).parent / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_verbose_gaps(numpy_audit, numpyts_audit):
    """Print detailed list of missing and extra functions."""
    numpy_toplevel = set()
    for funcs in numpy_audit['categorized'].values():
        numpy_toplevel.update(funcs)

    numpyts_toplevel = set(numpyts_audit['all_functions'].keys())
    numpy_methods = set(numpy_audit['ndarray_methods'].keys())
    numpyts_methods = set(numpyts_audit['ndarray_methods'].keys())

    print("\n" + "=" * 70)
    print("VERBOSE GAP ANALYSIS")
    print("=" * 70)

    # Top-level function gaps
    missing_toplevel = numpy_toplevel - numpyts_toplevel
    extra_toplevel = numpyts_toplevel - numpy_toplevel

    print("\nTOP-LEVEL FUNCTIONS:")
    print(f"  Missing from numpy-ts ({len(missing_toplevel)}):")
    for i, func in enumerate(sorted(missing_toplevel), 1):
        print(f"    {i:3d}. {func}")

    if extra_toplevel:
        print(f"\n  Extra in numpy-ts (not in NumPy, {len(extra_toplevel)}):")
        for i, func in enumerate(sorted(extra_toplevel), 1):
            print(f"    {i:3d}. {func}")

    # Method gaps
    missing_methods = numpy_methods - numpyts_methods
    extra_methods = numpyts_methods - numpy_methods

    print("\nNDARRAY METHODS:")
    print(f"  Missing from NDArray ({len(missing_methods)}):")
    for i, method in enumerate(sorted(missing_methods), 1):
        print(f"    {i:3d}. {method}")

    if extra_methods:
        print(f"\n  Extra in NDArray (not in ndarray, {len(extra_methods)}):")
        for i, method in enumerate(sorted(extra_methods), 1):
            print(f"    {i:3d}. {method}")

    print("\n" + "=" * 70)


def analyze_coverage(verbose=False):
    """Analyze API coverage between NumPy and numpy-ts."""
    numpy_audit = load_json('numpy-api-audit.json')
    numpyts_audit = load_json('numpyts-api-audit.json')

    # Check for uncategorized functions - FAIL if any exist
    uncategorized = numpy_audit.get('uncategorized', [])
    if uncategorized:
        print("\n" + "=" * 70)
        print("ERROR: UNCATEGORIZED FUNCTIONS FOUND")
        print("=" * 70)
        print(f"\n{len(uncategorized)} functions are not categorized in audit-numpy-api.py:")
        for i, func in enumerate(sorted(uncategorized)[:20], 1):
            print(f"  {i:3d}. {func}")
        if len(uncategorized) > 20:
            print(f"  ... and {len(uncategorized) - 20} more")
        print("\nAll NumPy functions must be categorized before compare-api can run.")
        print("Please add these functions to categories in scripts/audit-numpy-api.py")
        print("=" * 70)
        sys.exit(1)

    # Get ALL NumPy top-level functions (from categorized)
    numpy_toplevel = set()
    for funcs in numpy_audit['categorized'].values():
        numpy_toplevel.update(funcs)

    # Get ALL numpy-ts top-level functions
    numpyts_toplevel = set(numpyts_audit['all_functions'].keys())

    # Get methods
    numpy_methods = set(numpy_audit['ndarray_methods'].keys())
    numpyts_methods = set(numpyts_audit['ndarray_methods'].keys())

    print("=" * 70)
    print("ACCURATE API COVERAGE ANALYSIS")
    print("=" * 70)

    # Top-level comparison (ignoring if also methods)
    toplevel_implemented = numpy_toplevel & numpyts_toplevel
    toplevel_coverage = (
        100 * len(toplevel_implemented) / len(numpy_toplevel)
        if numpy_toplevel else 0
    )

    print("\nTOP-LEVEL FUNCTIONS (ignoring if also methods):")
    print(f"  NumPy functions:         {len(numpy_toplevel)}")
    print(f"  numpy-ts functions:      {len(numpyts_toplevel)}")
    print(f"  Implemented:             {len(toplevel_implemented)}")
    print(f"  Coverage:                "
          f"{len(toplevel_implemented)}/{len(numpy_toplevel)} "
          f"({toplevel_coverage:.1f}%)")

    # Methods comparison (ignoring if also functions)
    methods_implemented = numpy_methods & numpyts_methods
    methods_coverage = (
        100 * len(methods_implemented) / len(numpy_methods)
        if numpy_methods else 0
    )

    print("\nNDARRAY METHODS (ignoring if also functions):")
    print(f"  NumPy methods:           {len(numpy_methods)}")
    print(f"  numpy-ts methods:        {len(numpyts_methods)}")
    print(f"  Implemented:             {len(methods_implemented)}")
    print(f"  Coverage:                "
          f"{len(methods_implemented)}/{len(numpy_methods)} "
          f"({methods_coverage:.1f}%)")

    # Combined (for reference)
    # Total unique = union of all
    numpy_all = numpy_toplevel | numpy_methods
    numpyts_all = numpyts_toplevel | numpyts_methods

    # Count how many of NumPy's APIs are implemented in numpy-ts (ignore extras)
    implemented_apis = numpy_all & numpyts_all
    numpy_total = len(numpy_all)
    numpyts_implemented = len(implemented_apis)
    overall_coverage = 100 * numpyts_implemented / numpy_total if numpy_total else 0

    print("\nCOMBINED UNIQUE (functions ∪ methods):")
    print(f"  NumPy total:             {numpy_total}")
    print(f"  numpy-ts implemented:    {numpyts_implemented}")
    print(f"  Overall coverage:        "
          f"{numpyts_implemented}/{numpy_total} ({overall_coverage:.1f}%)")

    # Category breakdown
    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN (by top-level functions)")
    print("=" * 70)

    category_stats = {}
    # numpyts_all already defined above

    # Process all categories (including NDArray Methods)
    # Add NDArray Methods as a special category
    all_categories = {**numpy_audit['categorized'], 'NDArray Methods': sorted(numpy_methods)}

    for category, funcs in sorted(all_categories.items()):
        numpy_cat = set(funcs)

        # For NDArray Methods, check against numpyts_methods specifically
        if category == 'NDArray Methods':
            implemented = numpy_cat & numpyts_methods
        else:
            implemented = numpy_cat & numpyts_all

        pct = 100 * len(implemented) / len(numpy_cat) if numpy_cat else 0

        status = "✅" if pct == 100 else ("🟡" if pct >= 50 else "🔴")

        category_stats[category] = {
            'total': len(numpy_cat),
            'implemented': len(implemented),
            'missing': numpy_cat - (numpyts_methods if category == 'NDArray Methods' else numpyts_all),
            'percentage': pct,
            'status': status
        }

        impl_str = f"{len(implemented):3d}/{len(numpy_cat):3d}"
        pct_str = f"{pct:5.1f}%"
        print(f"{category:35s} {impl_str} ({pct_str}) {status}")

    # Print verbose gaps if requested
    if verbose:
        print_verbose_gaps(numpy_audit, numpyts_audit)

    return {
        'numpy_toplevel': len(numpy_toplevel),
        'numpyts_toplevel': len(numpyts_toplevel),
        'toplevel_coverage': toplevel_coverage,
        'numpy_methods': len(numpy_methods),
        'numpyts_methods': len(numpyts_methods),
        'methods_coverage': methods_coverage,
        'numpy_total': numpy_total,
        'numpyts_implemented': numpyts_implemented,
        'overall_coverage': overall_coverage,
        'category_stats': category_stats,
        'numpy_audit': numpy_audit,
        'numpyts_audit': numpyts_audit
    }


def generate_readme_table(category_stats):
    """Generate markdown table for README."""
    lines = []
    lines.append("| Category | Complete | Total | Status |")
    lines.append("|----------|----------|-------|--------|")

    # Sort by percentage (complete first)
    sorted_cats = sorted(
        category_stats.items(),
        key=lambda x: (-x[1]['percentage'], x[0])
    )

    for category, stats in sorted_cats:
        complete = f"{stats['implemented']}/{stats['total']}"
        pct = f"{stats['percentage']:.0f}%"
        status = stats['status']
        category_md = f"**{category}**"
        lines.append(f"| {category_md} | {complete} | {pct} | {status} |")

    return "\n".join(lines)


def update_readme(analysis):
    """Update README.md with accurate statistics."""
    readme_path = Path(__file__).parent.parent / 'README.md'

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    total_impl = analysis['numpyts_implemented']
    total_numpy = analysis['numpy_total']
    coverage = analysis['overall_coverage']

    # Update the badge
    # Determine color: green if >=90%, yellow if >=50%, red otherwise
    if coverage >= 90:
        color = 'brightgreen'
    elif coverage >= 50:
        color = 'yellow'
    else:
        color = 'red'

    # Update badge with proper URL encoding for percentage sign
    coverage_int = int(round(coverage))
    old_badge_pattern = r'!\[numpy api coverage\]\(https://img\.shields\.io/badge/numpy_api_coverage-\d+%20%25-\w+\)'
    new_badge = f'![numpy api coverage](https://img.shields.io/badge/numpy_api_coverage-{coverage_int}%20%25-{color})'
    content = re.sub(old_badge_pattern, new_badge, content)

    # Update the "Why numpy-ts?" section
    old_pattern = r'- \*\*📊 Extensive API\*\* — \*\*.*?\*\*'
    new_bullet = (
        f"- **📊 Extensive API** — **{total_impl} of {total_numpy} NumPy "
        f"functions ({coverage:.1f}% coverage)**"
    )
    content = re.sub(old_pattern, new_bullet, content)

    # Update the API Coverage table
    table = generate_readme_table(analysis['category_stats'])
    overall_line = (
        f"\n**Overall: {total_impl}/{total_numpy} "
        f"functions ({coverage:.1f}% coverage)**"
    )

    # Find and replace the table
    # Use a more specific pattern to avoid exponential backtracking
    table_pattern = (
        r'### API Coverage\n\n'
        r'Progress toward complete NumPy API compatibility:\n\n'
        r'(?:\|[^\n]*\n)+'  # Match table rows (non-capturing, possessive)
        r'\n?'  # Optional blank line before Overall
        r'\*\*Overall:[^\n]*\n'
    )
    new_table_section = (
        f"### API Coverage\n\n"
        f"Progress toward complete NumPy API compatibility:\n\n"
        f"{table}\n{overall_line}\n"
    )

    content = re.sub(table_pattern, new_table_section, content,
                     flags=re.DOTALL)

    # Update architecture diagram
    old_pattern = r'│  NumPy-Compatible API \(.*?\)   │'
    new_arch = f"│  NumPy-Compatible API ({total_impl}/{total_numpy})   │"
    content = re.sub(old_pattern, new_arch, content)

    # Update comparison table
    old_pattern = r'\| NumPy API Coverage \| .*? \|'
    new_comp = (
        f"| NumPy API Coverage | "
        f"{total_impl}/{total_numpy} ({coverage:.0f}%) |"
    )
    content = re.sub(old_pattern, new_comp, content)

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ Updated {readme_path}")


def update_package_json(analysis):
    """Update package.json description with current API coverage."""
    package_json_path = Path(__file__).parent.parent / 'package.json'

    with open(package_json_path, 'r', encoding='utf-8') as f:
        content = f.read()

    coverage = analysis['overall_coverage']
    coverage_int = int(round(coverage))

    # Update description with current coverage percentage
    old_pattern = r'"description": "Complete NumPy implementation for TypeScript and JavaScript \(\d+% API coverage\)"'
    new_description = f'"description": "Complete NumPy implementation for TypeScript and JavaScript ({coverage_int}% API coverage)"'

    content = re.sub(old_pattern, new_description, content)

    with open(package_json_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Updated {package_json_path} (coverage: {coverage_int}%)")


def update_api_reference(analysis):
    """Update API-REFERENCE.md with complete function list."""
    api_ref_path = Path(__file__).parent.parent / 'docs' / 'API-REFERENCE.md'

    # Read existing file to preserve Notes section
    notes_section = ""
    if api_ref_path.exists():
        with open(api_ref_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract everything from "## Notes" onwards
            notes_match = re.search(r'^## Notes\n.*', content, re.MULTILINE | re.DOTALL)
            if notes_match:
                notes_section = "\n" + notes_match.group(0)

    # Get current date
    from datetime import date
    today = date.today().strftime('%Y-%m-%d')

    # Get data
    numpy_audit = analysis['numpy_audit']
    numpyts_audit = analysis['numpyts_audit']
    category_stats = analysis['category_stats']

    # Get implemented set
    numpyts_all = set(numpyts_audit['all_functions'].keys()) | set(numpyts_audit['ndarray_methods'].keys())
    numpyts_methods = set(numpyts_audit['ndarray_methods'].keys())
    numpy_methods = set(numpy_audit['ndarray_methods'].keys())

    # Build content
    lines = []
    lines.append("# API Reference")
    lines.append("")
    lines.append("Complete NumPy 2.0+ API compatibility checklist.")
    lines.append("")
    lines.append(f"**Last Updated**: {today}")
    lines.append("")
    lines.append("## Progress Summary")
    lines.append("")
    lines.append("Based on `npm run compare-api`:")
    lines.append("")

    # Stats
    total_impl = analysis['numpyts_implemented']
    total_numpy = analysis['numpy_total']
    coverage = analysis['overall_coverage']

    lines.append(f"- **Overall Coverage**: {total_impl}/{total_numpy} ({coverage:.1f}%)")
    lines.append(f"- **Top-level Functions**: {analysis['numpyts_toplevel']}/{analysis['numpy_toplevel']} ({analysis['toplevel_coverage']:.1f}%)")
    lines.append(f"- **NDArray Methods**: {len(analysis['numpyts_audit']['ndarray_methods'])}/{analysis['numpy_methods']} ({analysis['methods_coverage']:.1f}%)")
    lines.append("")

    # Completed categories
    completed = [(cat, stats) for cat, stats in sorted(category_stats.items())
                 if stats['percentage'] == 100]
    incomplete = [(cat, stats) for cat, stats in sorted(category_stats.items())
                  if stats['percentage'] < 100]

    if completed:
        lines.append("### Completed Categories (100%)")
        for cat, stats in completed:
            lines.append(f"- {cat} ({stats['implemented']}/{stats['total']})")
        lines.append("")

    if incomplete:
        lines.append("### Incomplete Categories")
        for cat, stats in incomplete:
            lines.append(f"- {cat} ({stats['implemented']}/{stats['total']}) - {stats['percentage']:.1f}%")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Generate category sections (including NDArray Methods)
    for category in sorted(list(numpy_audit['categorized'].keys()) + ['NDArray Methods']):
        if category == 'NDArray Methods':
            funcs = sorted(numpy_methods)
            check_against = numpyts_methods
        else:
            funcs = sorted(numpy_audit['categorized'][category])
            check_against = numpyts_all

        lines.append(f"## {category}")
        lines.append("")

        for func in funcs:
            is_impl = func in check_against
            checkbox = "[x]" if is_impl else "[ ]"
            lines.append(f"- {checkbox} `{func}` ")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Add Extra NDArray Methods section
    extra_methods = sorted(numpyts_methods - set(numpy_methods))

    if extra_methods:
        lines.append("## Extra NDArray Methods")
        lines.append("")
        lines.append("Methods in numpy-ts NDArray that don't exist in NumPy's ndarray.")
        lines.append("These may be removed in future versions for strict NumPy compatibility:")
        lines.append("")

        for method in extra_methods:
            lines.append(f"- `{method}()` ")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Combine with notes section
    full_content = "\n".join(lines) + notes_section

    with open(api_ref_path, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"✅ Updated {api_ref_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare numpy-ts vs NumPy API and update documentation'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed list of missing and extra functions'
    )
    args = parser.parse_args()

    print("Comparing numpy-ts vs NumPy and updating documentation...\n")

    # Run audits first to get fresh data
    run_audits()

    # Analyze coverage
    analysis = analyze_coverage(verbose=args.verbose)

    # Update package.json
    update_package_json(analysis)

    # Update README
    update_readme(analysis)

    # Update API-REFERENCE.md
    update_api_reference(analysis)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal NumPy APIs: {analysis['numpy_total']}")
    print(f"Implemented in numpy-ts: {analysis['numpyts_implemented']}")
    print(f"Overall coverage: {analysis['overall_coverage']:.1f}%")

    print(f"\nTop-level functions: "
          f"{analysis['toplevel_coverage']:.1f}% coverage")
    print(f"NDArray methods: {analysis['methods_coverage']:.1f}% coverage")

    print("\nTop completing categories:")
    sorted_cats = sorted(
        analysis['category_stats'].items(),
        key=lambda x: -x[1]['percentage']
    )
    for cat, stats in sorted_cats[:5]:
        print(f"  {cat:30s} {stats['percentage']:5.1f}%")

    print("\nNext priorities (50-90% complete):")
    for cat, stats in sorted_cats:
        if 50 <= stats['percentage'] < 90:
            missing_list = ', '.join(sorted(stats['missing'])[:3])
            pct_str = f"{stats['percentage']:.0f}%"
            print(f"  {cat:30s} ({pct_str}) - Need: {missing_list}")

    print("\n" + "=" * 70)
    if args.verbose:
        print("Done! (use -v flag to see detailed gaps)")
    else:
        print("Done! (use -v flag for detailed missing/extra functions list)")


if __name__ == '__main__':
    main()
