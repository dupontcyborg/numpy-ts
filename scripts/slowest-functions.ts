/**
 * Show the N slowest functions by ratio (JS time / Python time).
 * Usage: tsx scripts/slowest-functions.ts [N=50]
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const sizeArg = process.argv.find((a: string) =>
  ['small', 'medium', 'large', 'pyodide'].includes(a),
);
const fileMap: Record<string, string> = {
  small: 'latest-full-small.json',
  medium: 'latest-full.json',
  large: 'latest-full-large.json',
  pyodide: 'latest-full-pyodide.json',
};
const resultFile = fileMap[sizeArg ?? 'medium'] ?? fileMap['medium']!;
const data = JSON.parse(
  readFileSync(join(__dirname, '../benchmarks/results', resultFile), 'utf-8'),
);

const N = parseInt(process.argv.find((a: string) => /^\d+$/.test(a)) ?? '50', 10);

interface Result {
  name: string;
  category: string;
  ratio: number;
  numpy: { mean_ms: number };
  numpyjs: { mean_ms: number };
}

// A benchmark that threw carries a zeroed timing, so its ratio is 0 — which
// would rank it as the *fastest* op in the suite. Exclude and report separately.
const failed: Result[] = (data.results as Result[]).filter(
  (r) => (r as { numpyjs?: { failed?: string } }).numpyjs?.failed,
);
const sorted: Result[] = [...data.results]
  .filter((r: Result) => !(r as { numpyjs?: { failed?: string } }).numpyjs?.failed)
  .filter((r: Result) => r.ratio != null && Number.isFinite(r.ratio))
  .sort((a: Result, b: Result) => b.ratio - a.ratio);

if (failed.length > 0) {
  console.log(`⚠  ${failed.length} benchmark(s) FAILED and are excluded (no measurement):`);
  for (const f of failed) {
    console.log(`     ${f.name}: ${(f as { numpyjs: { failed: string } }).numpyjs.failed}`);
  }
  console.log('');
}

console.log(`Top ${N} slowest functions (JS/Python ratio, higher = worse):\n`);
console.log(
  '#'.padStart(4) +
    '  ' +
    'Function'.padEnd(40) +
    'Category'.padEnd(16) +
    'Ratio'.padStart(8) +
    '  Python(ms)'.padStart(12) +
    '     JS(ms)'.padStart(12),
);
console.log('-'.repeat(94));

for (let i = 0; i < Math.min(N, sorted.length); i++) {
  const r = sorted[i]!;
  console.log(
    String(i + 1).padStart(4) +
      '  ' +
      r.name.padEnd(40) +
      r.category.padEnd(16) +
      r.ratio.toFixed(1).padStart(8) +
      ('  ' + r.numpy.mean_ms.toFixed(4)).padStart(12) +
      ('  ' + r.numpyjs.mean_ms.toFixed(4)).padStart(12),
  );
}
