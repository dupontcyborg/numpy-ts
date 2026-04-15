#!/usr/bin/env tsx
/**
 * Compare benchmark results: local (working tree) vs git (staged, HEAD, or main).
 *
 * Usage:
 *   tsx benchmarks/scripts/compare.ts [latest|latest-full] [--main]
 *
 * Defaults to "latest" if no argument given.
 * --main: compare against main branch HEAD instead of current branch's staged/HEAD.
 * Otherwise compares against staged version first; falls back to HEAD if nothing is staged.
 */

import { readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { resolve } from 'path';

const root = resolve(import.meta.dirname, '../..');
const args = process.argv.slice(2);
const useMain = args.includes('--main');
const positional = args.filter(a => !a.startsWith('--'));
const sizeMap: Record<string, string> = {
  small: 'latest-full-small',
  medium: 'latest-full',
  large: 'latest-full-large',
  pyodide: 'latest-full-pyodide',
};
const sizeArg = positional.find(a => a in sizeMap);
const variant = sizeArg ? sizeMap[sizeArg]! : positional[0] ?? 'latest-full';
const relPath = `benchmarks/results/${variant}.json`;
const localPath = resolve(root, relPath);

// --- Load local ---
if (!existsSync(localPath)) {
  console.error(`No local file: ${relPath}`);
  process.exit(1);
}
const local = JSON.parse(readFileSync(localPath, 'utf8'));

// --- Load git version ---
let gitJson: string | null = null;
let source = '';

if (useMain) {
  // Compare against main branch HEAD
  try {
    gitJson = execSync(`git show main:${relPath}`, { cwd: root, encoding: 'utf8', maxBuffer: 128 * 1024 * 1024, stdio: ['pipe', 'pipe', 'pipe'] });
    source = 'main';
  } catch {
    console.error(`No version of ${relPath} found on main branch.`);
    process.exit(1);
  }
} else {
  // Default: staged first, then HEAD
  try {
    // Try staged (:0 = index)
    gitJson = execSync(`git show :${relPath}`, { cwd: root, encoding: 'utf8', maxBuffer: 128 * 1024 * 1024, stdio: ['pipe', 'pipe', 'pipe'] });
    // Check if staged differs from HEAD (i.e. something is actually staged)
    try {
      execSync(`git diff --cached --quiet -- ${relPath}`, { cwd: root, stdio: 'pipe' });
      // No diff = nothing staged, fall through to HEAD
      gitJson = null;
    } catch {
      // diff --cached returned non-zero = there ARE staged changes
      // But if staged == local (working tree), comparing them is pointless — use HEAD instead
      try {
        execSync(`git diff --quiet -- ${relPath}`, { cwd: root, stdio: 'pipe' });
        // No diff = staged matches local, fall through to HEAD
        gitJson = null;
      } catch {
        source = 'staged';
      }
    }
  } catch {
    // Not in index at all
  }

  if (!gitJson) {
    try {
      gitJson = execSync(`git show HEAD:${relPath}`, { cwd: root, encoding: 'utf8', maxBuffer: 128 * 1024 * 1024, stdio: ['pipe', 'pipe', 'pipe'] });
      source = 'HEAD';
    } catch {
      console.error(`No git version of ${relPath} found (not staged or committed).`);
      process.exit(1);
    }
  }
}

const git = JSON.parse(gitJson);

// --- Build maps ---
interface BenchResult {
  name: string;
  ratio: number;
  ts_ms?: number;
  numpy_ms?: number;
  wasmUsed?: boolean;
}

const oldMap = new Map<string, BenchResult>();
for (const r of git.results as BenchResult[]) oldMap.set(r.name, r);

const newMap = new Map<string, BenchResult>();
for (const r of local.results as BenchResult[]) newMap.set(r.name, r);

// Append a WASM tag to a benchmark name based on old/new wasmUsed status:
//   [WASM]  - both runs used a WASM kernel
//   [+WASM] - new run added WASM (wasn't used before)
//   [-WASM] - old run had WASM, new run does not
const wasmTag = (name: string, oldUsed?: boolean, newUsed?: boolean): string => {
  if (oldUsed && newUsed) return ' [WASM]';
  if (!oldUsed && newUsed) return ' [+WASM]';
  if (oldUsed && !newUsed) return ' [-WASM]';
  return '';
};

const label = (name: string, oldUsed?: boolean, newUsed?: boolean) =>
  `${name}${wasmTag(name, oldUsed, newUsed)}`;

// --- Compare ---
interface Row {
  name: string;
  oldR: number;
  newR: number;
  pct: number;
}

const rows: Row[] = [];
for (const [name, newR] of newMap) {
  const oldR = oldMap.get(name);
  if (oldR != null) {
    const pct = (oldR.ratio - newR.ratio) / oldR.ratio * 100;
    rows.push({ name, oldR: oldR.ratio, newR: newR.ratio, pct });
  }
}
rows.sort((a, b) => b.pct - a.pct);

const pad = (s: string, n: number) => s.padStart(n);
const fmt = (n: number) => n.toFixed(2);

// Header
console.log(`\nComparing ${variant}.json: local vs ${source}`);
console.log(`  ${source}: ${git.timestamp}`);
console.log(`  local:  ${local.timestamp}`);
console.log(`  ${rows.length} benchmarks in common\n`);

// Improved
const improved = rows.filter(r => r.pct > 10);
if (improved.length > 0) {
  console.log(`IMPROVED (${improved.length}):`);
  for (const r of improved) {
    console.log(`  ${pad(fmt(r.oldR), 8)}x → ${pad(fmt(r.newR), 7)}x  ${pad(r.pct.toFixed(0) + '%', 5)}  ${label(r.name, oldMap.get(r.name)?.wasmUsed, newMap.get(r.name)?.wasmUsed)}`);
  }
  console.log();
}

// Regressed
const regressed = rows.filter(r => r.pct < -10);
if (regressed.length > 0) {
  console.log(`REGRESSED (${regressed.length}):`);
  for (const r of regressed) {
    console.log(`  ${pad(fmt(r.oldR), 8)}x → ${pad(fmt(r.newR), 7)}x  ${pad(r.pct.toFixed(0) + '%', 5)}  ${label(r.name, oldMap.get(r.name)?.wasmUsed, newMap.get(r.name)?.wasmUsed)}`);
  }
  console.log();
}

// Unchanged
const unchanged = rows.filter(r => r.pct >= -10 && r.pct <= 10);
if (unchanged.length > 0) {
  console.log(`UNCHANGED (${unchanged.length}):`);
  for (const r of unchanged) {
    console.log(`  ${pad(fmt(r.oldR), 8)}x → ${pad(fmt(r.newR), 7)}x  ${pad(r.pct.toFixed(0) + '%', 5)}  ${label(r.name, oldMap.get(r.name)?.wasmUsed, newMap.get(r.name)?.wasmUsed)}`);
  }
  console.log();
}

// Still slow (only overlapping benchmarks)
const slow = [...newMap.entries()].filter(([name, r]) => r.ratio > 10 && oldMap.has(name)).sort((a, b) => b[1].ratio - a[1].ratio);
if (slow.length > 0) {
  console.log(`STILL >10x SLOWER (${slow.length}):`);
  for (const [name] of slow) {
    const r = newMap.get(name)!;
    console.log(`  ${pad(r.ratio.toFixed(1), 7)}x  ${label(name, oldMap.get(name)?.wasmUsed, newMap.get(name)?.wasmUsed)}`);
  }
  console.log();
}

// New benchmarks (in local but not in git)
const added = [...newMap.keys()].filter(k => !oldMap.has(k));
if (added.length > 0) {
  console.log(`NEW (${added.length}):`);
  for (const name of added) {
    const r = newMap.get(name)!;
    console.log(`  ${pad(r.ratio.toFixed(2), 8)}x  ${label(name, undefined, r.wasmUsed)}`);
  }
  console.log();
}

// Removed benchmarks (in git but not in local)
const removed = [...oldMap.keys()].filter(k => !newMap.has(k));
if (removed.length > 0) {
  const REMOVED_LIMIT = 5;
  console.log(`REMOVED (${removed.length}):`);
  for (const name of removed.slice(0, REMOVED_LIMIT)) {
    console.log(`           ${label(name, oldMap.get(name)?.wasmUsed, undefined)}`);
  }
  if (removed.length > REMOVED_LIMIT) {
    console.log(`           ... and ${removed.length - REMOVED_LIMIT} more`);
  }
  console.log();
}

// Summary — use all local benchmarks for "new" side; overlapping only for "old" side
const allNewRatios = [...newMap.values()].map(r => r.ratio);
const geoMean = (arr: number[]) => Math.exp(arr.reduce((s, r) => s + Math.log(r), 0) / arr.length);
const geoOld = geoMean(rows.map(r => r.oldR));
const geoNew = geoMean(allNewRatios);
const medOld = rows.map(r => r.oldR).sort((a, b) => a - b)[Math.floor(rows.length / 2)]!;
const medNew = [...allNewRatios].sort((a, b) => a - b)[Math.floor(allNewRatios.length / 2)]!;

const wasmBoth    = rows.filter(r => oldMap.get(r.name)?.wasmUsed && newMap.get(r.name)?.wasmUsed).length;
const wasmAdded   = rows.filter(r => !oldMap.get(r.name)?.wasmUsed && newMap.get(r.name)?.wasmUsed).length;
const wasmRemoved = rows.filter(r => oldMap.get(r.name)?.wasmUsed && !newMap.get(r.name)?.wasmUsed).length;
console.log('SUMMARY:');
console.log(`  Geo mean:     ${fmt(geoOld)}x → ${fmt(geoNew)}x  (local: all ${allNewRatios.length} benchmarks)`);
console.log(`  Median ratio: ${fmt(medOld)}x → ${fmt(medNew)}x`);
console.log(`  Improved:     ${improved.length}  |  Regressed: ${regressed.length}  |  Unchanged: ${unchanged.length}`);
console.log(`  >10x slower:  ${slow.length}  (overlapping only)`);
if (wasmBoth + wasmAdded + wasmRemoved > 0) {
  const parts = [];
  if (wasmBoth)    parts.push(`[WASM] = both (${wasmBoth})`);
  if (wasmAdded)   parts.push(`[+WASM] = added (${wasmAdded})`);
  if (wasmRemoved) parts.push(`[-WASM] = removed (${wasmRemoved})`);
  console.log(`  ${parts.join('  ')}`);
}
console.log();
