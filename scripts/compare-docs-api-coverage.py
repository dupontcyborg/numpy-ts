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
from dataclasses import dataclass


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


@dataclass
class ParamSig:
    name: str
    type_text: str
    optional: bool
    rest: bool


@dataclass
class FuncSig:
    name: str
    params: list[ParamSig]
    return_type: str


def extract_docs_text(api_dir: Path) -> str:
    """Concatenate API docs text for lightweight mention checks."""
    parts: list[str] = []
    for path in api_dir.rglob('*.mdx'):
        parts.append(path.read_text(encoding='utf-8'))
    return "\n".join(parts)


def is_mentioned_in_docs(name: str, docs_text: str) -> bool:
    """True if a name appears as a standalone token in docs text."""
    return re.search(rf'(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])', docs_text) is not None


def split_top_level(s: str, sep: str = ',') -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = {'(': 0, '[': 0, '{': 0, '<': 0}
    close_to_open = {')': '(', ']': '[', '}': '{', '>': '<'}
    for ch in s:
        if ch in depth:
            depth[ch] += 1
        elif ch in close_to_open:
            o = close_to_open[ch]
            if depth[o] > 0:
                depth[o] -= 1
        if ch == sep and all(v == 0 for v in depth.values()):
            part = ''.join(cur).strip()
            if part:
                parts.append(part)
            cur = []
            continue
        cur.append(ch)
    tail = ''.join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def normalize_type_text(t: str) -> str:
    t = t.strip()
    # Canonicalize common public type aliases/wrapper internals.
    replacements = [
        (r'\bNDArrayCore\b', 'ArrayLike'),
        (r'\bNDArrayClass\b', 'ArrayLike'),
        (r'\bNDArray\b', 'ArrayLike'),
        (r'\bArrayStorage\b', 'ArrayLike'),
        (r'\bArrayInput\b', 'ArrayLike'),
        (r'\bunknown\b', 'ArrayLike'),
        (r'\bNestedArray\b', 'ArrayLike'),
        (r'\bTypedArray\b', 'ArrayLike'),
        (r'\bParseTxtOptionsType\b', 'ParseTxtOptions'),
        (r'\bNpzParseOptionsType\b', 'NpzParseOptions'),
        (r'\bNpyMetadataType\b', 'NpyMetadata'),
        (r'\bDTypeIO\b', 'DType'),
    ]
    for pat, repl in replacements:
        t = re.sub(pat, repl, t)
    t = t.replace('"', "'")

    # Collapse whitespace.
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()

    # Normalize function-type parameter names (keep their types/order).
    # Example: "(arr: ArrayLike, axis: number) => ArrayLike" ~= "(a: ArrayLike, axis: number) => ArrayLike"
    fn_pat = re.compile(r'\(([^()]*)\)\s*=>')
    def _fn_repl(m: re.Match[str]) -> str:
        params = m.group(1).strip()
        if not params:
            return '() =>'
        out_parts = []
        for p in split_top_level(params):
            p = p.strip()
            rest = p.startswith('...')
            if rest:
                p = p[3:].strip()
            if ':' in p:
                _, typ = p.split(':', 1)
                typ = typ.strip()
            else:
                typ = p
            out_parts.append(('...' if rest else '') + typ)
        return '(' + ', '.join(out_parts) + ') =>'
    t = fn_pat.sub(_fn_repl, t)

    # Normalize top-level unions order-insensitively.
    if '|' in t and '=>' not in t:
        parts = [p.strip() for p in split_top_level(t, '|')]
        norm_parts = [p for p in parts if p]
        # If ArrayLike is present, redundant concrete collection aliases add no precision.
        if 'ArrayLike' in norm_parts:
            norm_parts = [p for p in norm_parts if p not in {'number[]', 'ArrayLike[]'}]
        t = ' | '.join(sorted(dict.fromkeys(norm_parts)))

    return t


def parse_params(params_text: str) -> list[ParamSig]:
    if not params_text.strip():
        return []
    out: list[ParamSig] = []
    for p in split_top_level(params_text):
        p = p.strip()
        rest = p.startswith('...')
        if rest:
            p = p[3:].strip()
        if ':' not in p:
            # fallback
            name = p.rstrip('?').strip()
            optional = p.endswith('?')
            out.append(ParamSig(name=name, type_text='unknown', optional=optional, rest=rest))
            continue
        name_part, type_part = p.split(':', 1)
        name_part = name_part.strip()
        optional = name_part.endswith('?')
        name = name_part[:-1] if optional else name_part
        out.append(
            ParamSig(
                name=name.strip(),
                type_text=normalize_type_text(type_part),
                optional=optional,
                rest=rest,
            )
        )
    return out


def parse_func_signature(name: str, params_text: str, return_text: str) -> FuncSig:
    return FuncSig(
        name=name,
        params=parse_params(params_text),
        return_type=normalize_type_text(return_text),
    )


def extract_doc_signatures(api_dir: Path) -> dict[str, FuncSig]:
    sigs: dict[str, FuncSig] = {}
    pat = re.compile(r'function\s+([A-Za-z_][A-Za-z0-9_\.]*)\s*\(')
    for path in api_dir.rglob('*.mdx'):
        rel = path.relative_to(api_dir).as_posix()
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            m = pat.search(lines[i])
            if not m:
                i += 1
                continue
            name = m.group(1)
            if name == 'functionName':
                i += 1
                continue
            # capture until closing paren and return type
            chunk = lines[i]
            j = i + 1
            while j < len(lines) and ')' not in chunk:
                chunk += '\n' + lines[j]
                j += 1
            # include likely multiline return type lines
            while j < len(lines) and '```' not in lines[j] and not lines[j].strip().startswith('|'):
                if lines[j].strip() == '':
                    break
                chunk += '\n' + lines[j]
                j += 1
            full = chunk
            # parse name/params/return with paren matching
            name_start = full.find(name)
            paren_start = full.find('(', name_start)
            if paren_start == -1:
                i = j
                continue
            depth = 0
            paren_end = -1
            for k in range(paren_start, len(full)):
                c = full[k]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        paren_end = k
                        break
            if paren_end == -1:
                i = j
                continue
            params_text = full[paren_start + 1:paren_end]
            tail = full[paren_end + 1:].strip()
            if ':' not in tail:
                i = j
                continue
            return_text = tail.split(':', 1)[1].strip().rstrip(';')
            norm_name = normalize_doc_name(name, rel)
            sigs[norm_name] = parse_func_signature(norm_name, params_text, return_text)
            i = j
    return sigs


def extract_impl_signatures_from_dts() -> dict[str, FuncSig]:
    sigs: dict[str, FuncSig] = {}
    types_root = REPO_ROOT / 'dist/types'
    files = sorted(types_root.rglob('*.d.ts'))
    if not files:
        return sigs

    unresolved_aliases: list[tuple[str, str]] = []
    unresolved_namespace_refs: list[tuple[str, str]] = []

    def find_matching(text: str, start: int, open_ch: str = '(', close_ch: str = ')') -> int:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def find_return_end(text: str, start: int) -> int:
        depth = {'(': 0, '[': 0, '{': 0, '<': 0}
        close_to_open = {')': '(', ']': '[', '}': '{', '>': '<'}
        i = start
        while i < len(text):
            ch = text[i]
            if ch in depth:
                depth[ch] += 1
            elif ch in close_to_open:
                o = close_to_open[ch]
                if depth[o] > 0:
                    depth[o] -= 1
            elif ch == ';' and all(v == 0 for v in depth.values()):
                return i
            i += 1
        return -1

    def parse_declared_functions(text: str) -> dict[str, FuncSig]:
        out: dict[str, FuncSig] = {}
        fn_start_pat = re.compile(r'export declare function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
        for m in fn_start_pat.finditer(text):
            name = m.group(1)
            paren_start = m.end() - 1
            paren_end = find_matching(text, paren_start, '(', ')')
            if paren_end == -1:
                continue
            j = paren_end + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] != ':':
                continue
            ret_start = j + 1
            while ret_start < len(text) and text[ret_start].isspace():
                ret_start += 1
            ret_end = find_return_end(text, ret_start)
            if ret_end == -1:
                continue
            params_text = text[paren_start + 1:paren_end]
            return_text = text[ret_start:ret_end].strip()
            out[name] = parse_func_signature(name, params_text, return_text)
        return out

    def parse_namespace_methods(block: str, ns: str) -> dict[str, FuncSig]:
        out: dict[str, FuncSig] = {}
        meth_start_pat = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\(', re.M)
        for m in meth_start_pat.finditer(block):
            meth = m.group(1)
            paren_start = m.end() - 1
            paren_end = find_matching(block, paren_start, '(', ')')
            if paren_end == -1:
                continue
            j = paren_end + 1
            while j < len(block) and block[j].isspace():
                j += 1
            if not block.startswith('=>', j):
                continue
            ret_start = j + 2
            while ret_start < len(block) and block[ret_start].isspace():
                ret_start += 1
            ret_end = find_return_end(block, ret_start)
            if ret_end == -1:
                continue
            params_text = block[paren_start + 1:paren_end]
            return_text = block[ret_start:ret_end].strip()
            nm = f'{ns}.{meth}'
            out[nm] = parse_func_signature(nm, params_text, return_text)
        return out

    def clone_sig(new_name: str, src: FuncSig) -> FuncSig:
        return FuncSig(
            name=new_name,
            params=[
                ParamSig(
                    name=p.name,
                    type_text=p.type_text,
                    optional=p.optional,
                    rest=p.rest,
                )
                for p in src.params
            ],
            return_type=src.return_type,
        )

    def resolve_sig_ref(ref: str) -> FuncSig | None:
        # Try exact match first, then fall back to the terminal symbol.
        direct = sigs.get(ref)
        if direct is not None:
            return direct
        tail = ref.split('.')[-1]
        return sigs.get(tail)

    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding='utf-8')
        # top-level function declarations
        sigs.update(parse_declared_functions(text))

        # top-level const aliases to functions (e.g. "export declare const abs: typeof absolute;")
        alias_pat = re.compile(
            r'export declare const\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*typeof\s+([A-Za-z0-9_\.]+)\s*;'
        )
        for m in alias_pat.finditer(text):
            alias = m.group(1)
            target = m.group(2)
            target_sig = resolve_sig_ref(target)
            if target_sig is not None:
                sigs[alias] = clone_sig(alias, target_sig)
            else:
                unresolved_aliases.append((alias, target))

        # Parse object-namespace consts
        for ns in ('linalg', 'random', 'fft'):
            start_pat = f'export declare const {ns}: {{'
            start = text.find(start_pat)
            if start == -1:
                continue
            block_start = text.find('{', start)
            if block_start == -1:
                continue
            depth = 0
            block_end = -1
            for i, ch in enumerate(text[block_start:], start=block_start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        block_end = i
                        break
            if block_end == -1:
                continue
            block = text[block_start + 1:block_end]
            sigs.update(parse_namespace_methods(block, ns))
            meth_ref_pat = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*typeof\s+([A-Za-z0-9_\.]+)\s*;', re.M)
            for mm in meth_ref_pat.finditer(block):
                nm = f'{ns}.{mm.group(1)}'
                ref = mm.group(2)
                target_sig = resolve_sig_ref(ref)
                if target_sig is not None:
                    sigs[nm] = clone_sig(nm, target_sig)
                else:
                    unresolved_namespace_refs.append((nm, ref))

    # Retry unresolved refs after scanning all files.
    for alias, target in unresolved_aliases:
        target_sig = resolve_sig_ref(target)
        if target_sig is not None:
            sigs[alias] = clone_sig(alias, target_sig)
    for nm, ref in unresolved_namespace_refs:
        target_sig = resolve_sig_ref(ref)
        if target_sig is not None:
            sigs[nm] = clone_sig(nm, target_sig)

    return sigs


def compare_signatures(doc_sig: FuncSig, impl_sig: FuncSig) -> str | None:
    if len(doc_sig.params) != len(impl_sig.params):
        return f'param count differs: docs={len(doc_sig.params)} impl={len(impl_sig.params)}'
    for i, (dp, ip) in enumerate(zip(doc_sig.params, impl_sig.params), start=1):
        if dp.name != ip.name:
            return f'param {i} name differs: docs={dp.name} impl={ip.name}'
        if dp.optional != ip.optional:
            return f'param {i} optional differs: docs={dp.optional} impl={ip.optional}'
        if dp.rest != ip.rest:
            return f'param {i} rest differs: docs={dp.rest} impl={ip.rest}'
        if dp.type_text != ip.type_text:
            return f'param {i} type differs: docs={dp.type_text} impl={ip.type_text}'
    if doc_sig.return_type != impl_sig.return_type:
        return f'return type differs: docs={doc_sig.return_type} impl={impl_sig.return_type}'
    return None


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
    doc_signatures = extract_doc_signatures(api_dir)
    impl_signatures = extract_impl_signatures_from_dts()

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

    # Signature checks (default-on): only for doc functions with resolvable impl signatures.
    sign_mismatches: list[dict[str, str]] = []
    checked_signatures = 0
    uncheckable_doc_signatures = sorted(set(doc_signatures.keys()) - set(impl_signatures.keys()))
    for fn in sorted(set(doc_signatures.keys()) & set(impl_signatures.keys())):
        checked_signatures += 1
        err = compare_signatures(doc_signatures[fn], impl_signatures[fn])
        if err:
            sign_mismatches.append({'function': fn, 'reason': err})

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
        'signature_checks': {
            'checked': checked_signatures,
            'mismatches': sign_mismatches,
            'uncheckable_doc_signatures': uncheckable_doc_signatures,
        },
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
    print(f"Signature checks: {checked_signatures} checked, {len(sign_mismatches)} mismatches, {len(uncheckable_doc_signatures)} uncheckable")
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
        if sign_mismatches:
            print("\nSignature mismatches:")
            for i, sm in enumerate(sign_mismatches, 1):
                print(f"  {i:3d}. {sm['function']}: {sm['reason']}")
        if uncheckable_doc_signatures:
            print("\nDoc signatures without resolvable implementation signature:")
            for i, name in enumerate(uncheckable_doc_signatures[:200], 1):
                print(f"  {i:3d}. {name}")
    else:
        if missing_from_docs:
            print("\nTop missing from docs:")
            for name in missing_from_docs[:25]:
                print(f"  - {name}")
            if len(missing_from_docs) > 25:
                print(f"  ... and {len(missing_from_docs) - 25} more")

    # Exit non-zero if there are undocumented implemented functions or signature mismatches.
    sys.exit(1 if missing_from_docs or sign_mismatches else 0)


if __name__ == '__main__':
    main()
