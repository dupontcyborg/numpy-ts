#!/usr/bin/env tsx
/**
 * Compare benchmark results: local (working tree) vs git (staged or HEAD).
 *
 * Usage:
 *   tsx benchmarks/scripts/compare.ts [latest|latest-full]
 *
 * Defaults to "latest" if no argument given.
 * Compares against staged version first; falls back to HEAD if nothing is staged.
 */

import { readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { resolve } from 'path';

const root = resolve(import.meta.dirname, '../..');
const variant = process.argv[2] === 'latest-full' ? 'latest-full' : 'latest';
const relPath = `benchmarks/results/${variant}.json`;
const localPath = resolve(root, relPath);

// --- Load local ---
if (!existsSync(localPath)) {
  console.error(`No local file: ${relPath}`);
  process.exit(1);
}
const local = JSON.parse(readFileSync(localPath, 'utf8'));

// --- Load git version (staged first, then HEAD) ---
let gitJson: string | null = null;
let source = '';

try {
  // Try staged (:0 = index)
  gitJson = execSync(`git show :${relPath}`, { cwd: root, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
  // Check if staged differs from HEAD (i.e. something is actually staged)
  try {
    execSync(`git diff --cached --quiet -- ${relPath}`, { cwd: root, stdio: 'pipe' });
    // No diff = nothing staged, fall through to HEAD
    gitJson = null;
  } catch {
    // diff --cached returned non-zero = there ARE staged changes
    source = 'staged';
  }
} catch {
  // Not in index at all
}

if (!gitJson) {
  try {
    gitJson = execSync(`git show HEAD:${relPath}`, { cwd: root, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
    source = 'HEAD';
  } catch {
    console.error(`No git version of ${relPath} found (not staged or committed).`);
    process.exit(1);
  }
}

const git = JSON.parse(gitJson);

// --- Build maps ---
interface BenchResult {
  name: string;
  ratio: number;
  ts_ms?: number;
  numpy_ms?: number;
}

const oldMap = new Map<string, BenchResult>();
for (const r of git.results as BenchResult[]) oldMap.set(r.name, r);

const newMap = new Map<string, BenchResult>();
for (const r of local.results as BenchResult[]) newMap.set(r.name, r);

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
    console.log(`  ${pad(fmt(r.oldR), 8)}x → ${pad(fmt(r.newR), 7)}x  ${pad(r.pct.toFixed(0) + '%', 5)}  ${r.name}`);
  }
  console.log();
}

// Regressed
const regressed = rows.filter(r => r.pct < -10);
if (regressed.length > 0) {
  console.log(`REGRESSED (${regressed.length}):`);
  for (const r of regressed) {
    console.log(`  ${pad(fmt(r.oldR), 8)}x → ${pad(fmt(r.newR), 7)}x  ${pad(r.pct.toFixed(0) + '%', 5)}  ${r.name}`);
  }
  console.log();
}

// Unchanged
const unchanged = rows.filter(r => r.pct >= -10 && r.pct <= 10);
if (unchanged.length > 0) {
  console.log(`UNCHANGED (${unchanged.length}):`);
  for (const r of unchanged) {
    console.log(`  ${pad(fmt(r.oldR), 8)}x → ${pad(fmt(r.newR), 7)}x  ${pad(r.pct.toFixed(0) + '%', 5)}  ${r.name}`);
  }
  console.log();
}

// Still slow
const slow = [...newMap.entries()].filter(([, r]) => r.ratio > 10).sort((a, b) => b[1].ratio - a[1].ratio);
if (slow.length > 0) {
  console.log(`STILL >10x SLOWER (${slow.length}):`);
  for (const [name, r] of slow) {
    console.log(`  ${pad(r.ratio.toFixed(1), 7)}x  ${name}`);
  }
  console.log();
}

// New benchmarks (in local but not in git)
const added = [...newMap.keys()].filter(k => !oldMap.has(k));
if (added.length > 0) {
  console.log(`NEW (${added.length}):`);
  for (const name of added) {
    const r = newMap.get(name)!;
    console.log(`  ${pad(r.ratio.toFixed(2), 8)}x  ${name}`);
  }
  console.log();
}

// Removed benchmarks (in git but not in local)
const removed = [...oldMap.keys()].filter(k => !newMap.has(k));
if (removed.length > 0) {
  console.log(`REMOVED (${removed.length}):`);
  for (const name of removed) {
    console.log(`           ${name}`);
  }
  console.log();
}

// Summary
const avgOld = rows.reduce((s, r) => s + r.oldR, 0) / rows.length;
const avgNew = rows.reduce((s, r) => s + r.newR, 0) / rows.length;
const medOld = rows.map(r => r.oldR).sort((a, b) => a - b)[Math.floor(rows.length / 2)]!;
const medNew = rows.map(r => r.newR).sort((a, b) => a - b)[Math.floor(rows.length / 2)]!;

console.log('SUMMARY:');
console.log(`  Mean ratio:   ${fmt(avgOld)}x → ${fmt(avgNew)}x`);
console.log(`  Median ratio: ${fmt(medOld)}x → ${fmt(medNew)}x`);
console.log(`  Improved:     ${improved.length}  |  Regressed: ${regressed.length}  |  Unchanged: ${unchanged.length}`);
console.log(`  >10x slower:  ${slow.length}`);
console.log();
