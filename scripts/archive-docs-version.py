#!/usr/bin/env python3
"""Maintain canonical and noindex frontmatter across the versioned docs tree.

The live docs are served from docs/latest/ under stable URLs. Archived version
folders and docs/next/ are near-identical copies, so each one has to tell search
engines which copy is authoritative. A page gets exactly one of the two markers,
never both: noindex is a directive Google obeys and canonical is a hint it may
ignore, so a page carrying both drops out of the index entirely and forwards
nothing to the live copy.

Run `check` in CI. Every other mode mutates frontmatter in place and is safe to
re-run.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
DOCS_JSON = DOCS / "docs.json"

SITE = "https://numpyts.dev"
LATEST = "latest"
NEXT = "next"

# Pages listed in every version's navigation but existing as a single file at a
# single URL. Version-scoping their canonical would point several versions at
# one page that is not a copy of any of them.
SHARED_PAGES = {"index", "changelog"}

VERSION_DIR = re.compile(r"^v\d+\.\d+\.x$")
MANAGED_KEYS = ("canonical", "noindex")

# Absolute links to a versioned asset, which are pinned per version by design.
ASSET_LINK = re.compile(r"/assets/[^\s\"')]+")
VERSION_LINK = re.compile(r"^/v\d+\.\d+\.x/")
ABS_LINK = re.compile(r"\]\((/[^)]+)\)|href=\"(/[^\"]+)\"")


# --------------------------------------------------------------------------
# Frontmatter
# --------------------------------------------------------------------------

def split_frontmatter(text: str) -> tuple[list[str], str]:
    """Return the frontmatter lines and the remaining body.

    Line-based rather than a YAML round-trip so that quoting, key order and
    blank lines in the other keys survive untouched.
    """
    if not text.startswith("---\n"):
        raise ValueError("no leading frontmatter")
    end = text.index("\n---", 3)
    return text[4:end].split("\n"), text[end + 4:]


def join_frontmatter(lines: list[str], body: str) -> str:
    return "---\n" + "\n".join(lines) + "\n---" + body


def read_key(lines: list[str], key: str) -> str | None:
    for line in lines:
        if line.startswith(f"{key}:"):
            return line[len(key) + 1:].strip()
    return None


def apply_keys(path: Path, **keys: str | None) -> bool:
    """Set or remove frontmatter keys. A None value removes the key."""
    text = path.read_text(encoding="utf-8")
    lines, body = split_frontmatter(text)
    for key, value in keys.items():
        lines = [ln for ln in lines if not ln.startswith(f"{key}:")]
        if value is not None:
            lines.append(f"{key}: {value}")
    updated = join_frontmatter(lines, body)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


# --------------------------------------------------------------------------
# Tree helpers
# --------------------------------------------------------------------------

def pages_in(folder: str) -> list[Path]:
    root = DOCS / folder
    if not root.is_dir():
        sys.exit(f"error: docs/{folder} does not exist")
    return sorted(root.rglob("*.mdx"))


def archived_versions() -> list[str]:
    return sorted(d.name for d in DOCS.iterdir() if d.is_dir() and VERSION_DIR.match(d.name))


def shared_page_files() -> set[Path]:
    return {DOCS / f"{name}.mdx" for name in SHARED_PAGES}


def redirect_sources() -> set[str]:
    config = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    return {r["source"] for r in config.get("redirects", [])}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_archive(version: str) -> int:
    """Point an archived version at its live equivalent, or hide it."""
    canonical = hidden = 0
    for page in pages_in(version):
        rel = page.relative_to(DOCS / version).with_suffix("")
        if (DOCS / LATEST / rel).with_suffix(".mdx").exists():
            apply_keys(page, canonical=f"{SITE}/{LATEST}/{rel.as_posix()}", noindex=None)
            canonical += 1
        else:
            # No live counterpart, so a canonical would point at a 404.
            apply_keys(page, noindex="true", canonical=None)
            hidden += 1
    print(f"{version}: {canonical} canonical -> /{LATEST}, {hidden} noindex (no counterpart)")
    return 0


def cmd_set_noindex(folder: str) -> int:
    for page in pages_in(folder):
        apply_keys(page, noindex="true", canonical=None)
    print(f"{folder}: {len(pages_in(folder))} pages noindex")
    return 0


def cmd_unset_noindex(folder: str) -> int:
    """Clear both markers so the folder is self-canonical and indexable."""
    changed = sum(apply_keys(p, noindex=None, canonical=None) for p in pages_in(folder))
    print(f"{folder}: cleared markers on {changed} pages ({len(pages_in(folder))} total)")
    return 0


def cmd_emit_nav(version: str, insert: bool) -> int:
    """Copy the live navigation block, rewriting its page paths to a version folder."""
    config = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    versions = config["navigation"]["versions"]

    def repath(node):
        if isinstance(node, dict):
            return {k: [repath(p) for p in v] if k == "pages" else repath(v) for k, v in node.items()}
        if isinstance(node, list):
            return [repath(i) for i in node]
        if isinstance(node, str) and node.startswith(f"{LATEST}/"):
            return f"{version}/" + node[len(LATEST) + 1:]
        return node

    block = repath(copy.deepcopy(versions[0]))
    block["version"] = version
    block.pop("tag", None)

    if not insert:
        print(json.dumps(block, indent=2))
        return 0

    if any(v.get("version") == version for v in versions):
        sys.exit(f"error: docs.json already has a {version} block")
    versions.insert(1, block)
    DOCS_JSON.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"inserted {version} navigation block into docs.json")
    return 0


def cmd_check() -> int:
    problems: list[str] = []
    shared = shared_page_files()
    allowed_abs = redirect_sources()

    def flag(page: Path, msg: str) -> None:
        problems.append(f"{page.relative_to(REPO_ROOT)}: {msg}")

    # Marker placement.
    for version in archived_versions():
        for page in pages_in(version):
            lines, _ = split_frontmatter(page.read_text(encoding="utf-8"))
            canonical = read_key(lines, "canonical")
            noindex = read_key(lines, "noindex")
            if canonical and noindex:
                flag(page, "has both canonical and noindex; they are conflicting signals")
            elif not canonical and not noindex:
                flag(page, "archived page has neither canonical nor noindex")
            if canonical:
                if not canonical.startswith(f"{SITE}/{LATEST}/"):
                    flag(page, f"canonical does not point into /{LATEST}: {canonical}")
                else:
                    target = DOCS / LATEST / (canonical[len(f"{SITE}/{LATEST}/"):] + ".mdx")
                    if not target.exists():
                        flag(page, f"canonical target does not exist: {canonical}")

    for page in pages_in(LATEST):
        lines, _ = split_frontmatter(page.read_text(encoding="utf-8"))
        if read_key(lines, "noindex"):
            flag(page, f"/{LATEST} must stay indexable")
        if read_key(lines, "canonical"):
            flag(page, f"/{LATEST} should be self-canonical; drop the explicit canonical")

    for page in pages_in(NEXT):
        lines, _ = split_frontmatter(page.read_text(encoding="utf-8"))
        if not read_key(lines, "noindex"):
            flag(page, f"/{NEXT} is a copy of /{LATEST} and must be noindex")

    for page in sorted(shared):
        lines, _ = split_frontmatter(page.read_text(encoding="utf-8"))
        if read_key(lines, "canonical") or read_key(lines, "noindex"):
            flag(page, "shared page must not carry a version-scoped marker")

    # Link and asset integrity across every page.
    for page in sorted(DOCS.rglob("*.mdx")):
        text = page.read_text(encoding="utf-8")
        for asset in ASSET_LINK.findall(text):
            if not (DOCS / asset.lstrip("/")).exists():
                flag(page, f"asset reference does not exist: {asset}")
        for match in ABS_LINK.findall(text):
            link = (match[0] or match[1]).split("#")[0].rstrip("/")
            if link.startswith("/assets/") or link.startswith(f"/{LATEST}/") or link in allowed_abs:
                continue
            # A changelog entry documents one release, so it links to the docs as
            # they stood for that release rather than to the moving live copy.
            if page in shared and VERSION_LINK.match(link):
                continue
            flag(page, f"absolute link pins a version or staging folder: {link}")

    # Navigation integrity.
    config = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    versions = config["navigation"]["versions"]

    def walk(node, acc):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "pages":
                    for p in value:
                        acc.append(p) if isinstance(p, str) else walk(p, acc)
                else:
                    walk(value, acc)
        elif isinstance(node, list):
            for item in node:
                walk(item, acc)

    for block in versions:
        acc: list[str] = []
        walk(block, acc)
        for entry in acc:
            if not (DOCS / f"{entry}.mdx").exists():
                problems.append(f"docs.json [{block['version']}]: page does not exist: {entry}")
    live: list[str] = []
    walk(versions[0], live)
    if not any(p.startswith(f"{LATEST}/") for p in live):
        problems.append(f"docs.json: the first version block must serve pages from {LATEST}/")

    for redirect in config.get("redirects", []):
        if not redirect["destination"].startswith(f"/{LATEST}/"):
            problems.append(f"docs.json: redirect {redirect['source']} does not point at /{LATEST}")

    if problems:
        print(f"docs check failed ({len(problems)} problem(s)):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("docs check passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify markers, links, assets and navigation")
    p = sub.add_parser("archive", help="canonicalize or hide an archived version folder")
    p.add_argument("version")
    p = sub.add_parser("set-noindex", help="mark every page in a folder noindex")
    p.add_argument("folder")
    p = sub.add_parser("unset-noindex", help="clear markers so a folder is indexable")
    p.add_argument("folder")
    p = sub.add_parser("emit-nav", help="build a docs.json navigation block for a version")
    p.add_argument("version")
    p.add_argument("--insert", action="store_true", help="splice into docs.json instead of printing")

    args = parser.parse_args()
    if args.command == "check":
        return cmd_check()
    if args.command == "archive":
        return cmd_archive(args.version)
    if args.command == "set-noindex":
        return cmd_set_noindex(args.folder)
    if args.command == "unset-noindex":
        return cmd_unset_noindex(args.folder)
    if args.command == "emit-nav":
        return cmd_emit_nav(args.version, args.insert)
    return 1


if __name__ == "__main__":
    sys.exit(main())
