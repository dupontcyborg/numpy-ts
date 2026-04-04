/**
 * Show the N slowest functions by ratio (JS time / Python time).
 * Usage: tsx scripts/slowest-functions.ts [N=50]
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(
  readFileSync(join(__dirname, '../benchmarks/results/latest-full-large.json'), 'utf-8')
);

const N = parseInt(process.argv[2] ?? '50', 10);

interface Result {
  name: string;
  category: string;
  ratio: number;
  numpy: { mean_ms: number };
  numpyjs: { mean_ms: number };
}

const sorted: Result[] = [...data.results]
  .filter((r: Result) => r.ratio != null && isFinite(r.ratio))
  .sort((a: Result, b: Result) => b.ratio - a.ratio);

console.log(`Top ${N} slowest functions (JS/Python ratio, higher = worse):\n`);
console.log(
  '#'.padStart(4) +
    '  ' +
    'Function'.padEnd(40) +
    'Category'.padEnd(16) +
    'Ratio'.padStart(8) +
    '  Python(ms)'.padStart(12) +
    '     JS(ms)'.padStart(12)
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
      ('  ' + r.numpyjs.mean_ms.toFixed(4)).padStart(12)
  );
}
