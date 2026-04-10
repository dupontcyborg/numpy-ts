/**
 * Check that every exported Zig function is called at least once in a test block.
 * Usage: npx tsx scripts/zig-fn-coverage.ts [--verbose]
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

// --- Config ---
const THRESHOLD = 95; // minimum % to pass
const ZIG_DIR = 'src/common/wasm/zig';
const SKIP_FILES = new Set(['simd.zig', 'alloc.zig', 'sorting_common.zig', 'ziggurat_tables.zig']);

const verbose = process.argv.includes('--verbose');
const list = process.argv.includes('--list');
const files = readdirSync(ZIG_DIR)
  .filter((f) => f.endsWith('.zig') && !SKIP_FILES.has(f))
  .sort();

interface FileStat {
  file: string;
  total: number;
  covered: number;
  uncoveredFns: string[];
}

const stats: FileStat[] = [];

for (const file of files) {
  const src = readFileSync(join(ZIG_DIR, file), 'utf-8');
  const exportFns = [...src.matchAll(/^export fn (\w+)/gm)].map((m) => m[1]);
  if (exportFns.length === 0) continue;

  const testStart = src.search(/^test\s+"/m);
  const testSection = testStart >= 0 ? src.slice(testStart) : '';

  const stat: FileStat = { file, total: exportFns.length, covered: 0, uncoveredFns: [] };

  for (const fn of exportFns) {
    const re = new RegExp(`\\b${fn}\\b`);
    if (testSection && re.test(testSection)) {
      stat.covered++;
    } else {
      stat.uncoveredFns.push(fn);
    }
  }
  stats.push(stat);
}

const totalFns = stats.reduce((s, f) => s + f.total, 0);
const coveredFns = stats.reduce((s, f) => s + f.covered, 0);
const pct = totalFns > 0 ? (coveredFns / totalFns) * 100 : 0;

// --- Vitest-style table output ---
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[0m`;

const colorPct = (v: number) => {
  const s = v.toFixed(1).padStart(6) + '%';
  if (v >= 80) return green(s);
  if (v >= THRESHOLD) return yellow(s);
  return red(s);
};

const fileCol = 40;
const numCol = 8;
const sep = dim('-'.repeat(fileCol + numCol * 3 + 6));

console.log();
console.log(bold(' Zig export fn test coverage'));
console.log(sep);
console.log(
  ' ' +
    'File'.padEnd(fileCol) +
    dim('Fns'.padStart(numCol)) +
    dim('Covered'.padStart(numCol)) +
    dim('Missing'.padStart(numCol)) +
    '  ' +
    dim('Coverage')
);
console.log(sep);

const filesToShow = verbose ? stats : stats.filter((f) => f.covered < f.total);

for (const f of filesToShow) {
  const p = f.total > 0 ? (f.covered / f.total) * 100 : 100;
  const missing = f.total - f.covered;
  console.log(
    ' ' +
      f.file.padEnd(fileCol) +
      String(f.total).padStart(numCol) +
      String(f.covered).padStart(numCol) +
      (missing > 0 ? red(String(missing).padStart(numCol)) : String(0).padStart(numCol)) +
      '  ' +
      colorPct(p)
  );
}

if (!verbose && filesToShow.length < stats.length) {
  const hidden = stats.length - filesToShow.length;
  console.log(dim(`  ... ${hidden} files with 100% coverage (use --verbose to show all)`));
}

console.log(sep);
console.log(
  bold(' All files'.padEnd(fileCol + 1)) +
    bold(String(totalFns).padStart(numCol)) +
    bold(String(coveredFns).padStart(numCol)) +
    bold(String(totalFns - coveredFns).padStart(numCol)) +
    '  ' +
    colorPct(pct)
);
console.log();

// --- List uncovered functions ---
if (list) {
  const allUncovered = stats.flatMap((f) => f.uncoveredFns.map((fn) => ({ file: f.file, fn })));
  if (allUncovered.length > 0) {
    console.log(bold(' Uncovered functions:'));
    for (const { file, fn } of allUncovered) {
      console.log(`  ${dim(file + ':')} ${fn}`);
    }
    console.log();
  }
}

// --- Threshold check ---
if (pct < THRESHOLD) {
  console.log(red(`  Coverage ${pct.toFixed(1)}% is below threshold ${THRESHOLD}%`));
  console.log();
  process.exit(1);
} else {
  console.log(green(`  Coverage ${pct.toFixed(1)}% meets threshold ${THRESHOLD}%`));
  console.log();
}
